Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c runtime.bat", 0, False
Set WshShell = Nothing