windows_link::link!("kernel32.dll" "system" fn AcquireSRWLockExclusive(srwlock : *mut RTL_SRWLOCK));
windows_link::link!("kernel32.dll" "system" fn AcquireSRWLockShared(srwlock : *mut RTL_SRWLOCK));
windows_link::link!("kernel32.dll" "system" fn AddVectoredExceptionHandler(first : u32, handler : PVECTORED_EXCEPTION_HANDLER) -> *mut core::ffi::c_void);
windows_link::link!("kernel32.dll" "system" fn CancelIo(hfile : HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CloseHandle(hobject : HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CompareStringOrdinal(lpstring1 : *const u16, cchcount1 : i32, lpstring2 : *const u16, cchcount2 : i32, bignorecase : BOOL) -> i32);
windows_link::link!("kernel32.dll" "system" fn CopyFileExW(lpexistingfilename : PCWSTR, lpnewfilename : PCWSTR, lpprogressroutine : LPPROGRESS_ROUTINE, lpdata : *const core::ffi::c_void, pbcancel : *mut BOOL, dwcopyflags : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CreateDirectoryW(lppathname : PCWSTR, lpsecurityattributes : *const SECURITY_ATTRIBUTES) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CreateEventW(lpeventattributes : *const SECURITY_ATTRIBUTES, bmanualreset : BOOL, binitialstate : BOOL, lpname : PCWSTR) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn CreateFileW(lpfilename : PCWSTR, dwdesiredaccess : u32, dwsharemode : u32, lpsecurityattributes : *const SECURITY_ATTRIBUTES, dwcreationdisposition : u32, dwflagsandattributes : u32, htemplatefile : HANDLE) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn CreateHardLinkW(lpfilename : PCWSTR, lpexistingfilename : PCWSTR, lpsecurityattributes : *const SECURITY_ATTRIBUTES) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CreateNamedPipeW(lpname : PCWSTR, dwopenmode : u32, dwpipemode : u32, nmaxinstances : u32, noutbuffersize : u32, ninbuffersize : u32, ndefaulttimeout : u32, lpsecurityattributes : *const SECURITY_ATTRIBUTES) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn CreatePipe(hreadpipe : *mut HANDLE, hwritepipe : *mut HANDLE, lppipeattributes : *const SECURITY_ATTRIBUTES, nsize : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CreateProcessW(lpapplicationname : PCWSTR, lpcommandline : PWSTR, lpprocessattributes : *const SECURITY_ATTRIBUTES, lpthreadattributes : *const SECURITY_ATTRIBUTES, binherithandles : BOOL, dwcreationflags : u32, lpenvironment : *const core::ffi::c_void, lpcurrentdirectory : PCWSTR, lpstartupinfo : *const STARTUPINFOW, lpprocessinformation : *mut PROCESS_INFORMATION) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn CreateSymbolicLinkW(lpsymlinkfilename : PCWSTR, lptargetfilename : PCWSTR, dwflags : u32) -> bool);
windows_link::link!("kernel32.dll" "system" fn CreateThread(lpthreadattributes : *const SECURITY_ATTRIBUTES, dwstacksize : usize, lpstartaddress : LPTHREAD_START_ROUTINE, lpparameter : *const core::ffi::c_void, dwcreationflags : u32, lpthreadid : *mut u32) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn CreateWaitableTimerExW(lptimerattributes : *const SECURITY_ATTRIBUTES, lptimername : PCWSTR, dwflags : u32, dwdesiredaccess : u32) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn DeleteFileW(lpfilename : PCWSTR) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn DeleteProcThreadAttributeList(lpattributelist : *mut _PROC_THREAD_ATTRIBUTE_LIST));
windows_link::link!("kernel32.dll" "system" fn DeviceIoControl(hdevice : HANDLE, dwiocontrolcode : u32, lpinbuffer : *const core::ffi::c_void, ninbuffersize : u32, lpoutbuffer : *mut core::ffi::c_void, noutbuffersize : u32, lpbytesreturned : *mut u32, lpoverlapped : *mut OVERLAPPED) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn DuplicateHandle(hsourceprocesshandle : HANDLE, hsourcehandle : HANDLE, htargetprocesshandle : HANDLE, lptargethandle : *mut HANDLE, dwdesiredaccess : u32, binherithandle : BOOL, dwoptions : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn ExitProcess(uexitcode : u32) -> !);
windows_link::link!("kernel32.dll" "system" fn FindClose(hfindfile : HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn FindFirstFileExW(lpfilename : PCWSTR, finfolevelid : FINDEX_INFO_LEVELS, lpfindfiledata : *mut core::ffi::c_void, fsearchop : FINDEX_SEARCH_OPS, lpsearchfilter : *const core::ffi::c_void, dwadditionalflags : u32) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn FindNextFileW(hfindfile : HANDLE, lpfindfiledata : *mut WIN32_FIND_DATAW) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn FlsAlloc(lpcallback : PFLS_CALLBACK_FUNCTION) -> u32);
windows_link::link!("kernel32.dll" "system" fn FlsFree(dwflsindex : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn FlsGetValue(dwflsindex : u32) -> *mut core::ffi::c_void);
windows_link::link!("kernel32.dll" "system" fn FlsSetValue(dwflsindex : u32, lpflsdata : *const core::ffi::c_void) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn FlushFileBuffers(hfile : HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn FormatMessageW(dwflags : u32, lpsource : *const core::ffi::c_void, dwmessageid : u32, dwlanguageid : u32, lpbuffer : PCWSTR, nsize : u32, arguments : *const va_list) -> u32);
windows_link::link!("kernel32.dll" "system" fn FreeEnvironmentStringsW(penv : *const u16) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetActiveProcessorCount(groupnumber : u16) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetCommandLineW() -> PWSTR);
windows_link::link!("kernel32.dll" "system" fn GetConsoleMode(hconsolehandle : HANDLE, lpmode : *mut u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetConsoleOutputCP() -> u32);
windows_link::link!("kernel32.dll" "system" fn GetCurrentDirectoryW(nbufferlength : u32, lpbuffer : PWSTR) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetCurrentProcess() -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn GetCurrentProcessId() -> u32);
windows_link::link!("kernel32.dll" "system" fn GetCurrentThread() -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn GetCurrentThreadId() -> u32);
windows_link::link!("kernel32.dll" "system" fn GetEnvironmentStringsW() -> LPWCH);
windows_link::link!("kernel32.dll" "system" fn GetEnvironmentVariableW(lpname : PCWSTR, lpbuffer : PWSTR, nsize : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetExitCodeProcess(hprocess : HANDLE, lpexitcode : *mut u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetFileAttributesW(lpfilename : PCWSTR) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetFileInformationByHandle(hfile : HANDLE, lpfileinformation : *mut BY_HANDLE_FILE_INFORMATION) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetFileInformationByHandleEx(hfile : HANDLE, fileinformationclass : FILE_INFO_BY_HANDLE_CLASS, lpfileinformation : *mut core::ffi::c_void, dwbuffersize : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetFileSizeEx(hfile : HANDLE, lpfilesize : *mut i64) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetFileType(hfile : HANDLE) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetFinalPathNameByHandleW(hfile : HANDLE, lpszfilepath : PWSTR, cchfilepath : u32, dwflags : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetFullPathNameW(lpfilename : PCWSTR, nbufferlength : u32, lpbuffer : PWSTR, lpfilepart : *mut PWSTR) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetLastError() -> u32);
windows_link::link!("kernel32.dll" "system" fn GetModuleFileNameW(hmodule : HMODULE, lpfilename : PWSTR, nsize : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetModuleHandleA(lpmodulename : PCSTR) -> HMODULE);
windows_link::link!("kernel32.dll" "system" fn GetModuleHandleExW(dwflags : u32, lpmodulename : PCWSTR, phmodule : *mut HMODULE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetModuleHandleW(lpmodulename : PCWSTR) -> HMODULE);
windows_link::link!("kernel32.dll" "system" fn GetOverlappedResult(hfile : HANDLE, lpoverlapped : *const OVERLAPPED, lpnumberofbytestransferred : *mut u32, bwait : BOOL) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetProcAddress(hmodule : HMODULE, lpprocname : PCSTR) -> FARPROC);
windows_link::link!("kernel32.dll" "system" fn GetProcessId(process : HANDLE) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetStdHandle(nstdhandle : u32) -> HANDLE);
windows_link::link!("kernel32.dll" "system" fn GetSystemDirectoryW(lpbuffer : PWSTR, usize : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn GetSystemInfo(lpsysteminfo : *mut SYSTEM_INFO));
windows_link::link!("kernel32.dll" "system" fn GetSystemTimeAsFileTime(lpsystemtimeasfiletime : *mut FILETIME));
windows_link::link!("kernel32.dll" "system" fn GetSystemTimePreciseAsFileTime(lpsystemtimeasfiletime : *mut FILETIME));
windows_link::link!("kernel32.dll" "system" fn GetTempPathW(nbufferlength : u32, lpbuffer : PWSTR) -> u32);
windows_link::link!("userenv.dll" "system" fn GetUserProfileDirectoryW(htoken : HANDLE, lpprofiledir : PWSTR, lpcchsize : *mut u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn GetWindowsDirectoryW(lpbuffer : PWSTR, usize : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn InitOnceBeginInitialize(lpinitonce : LPINIT_ONCE, dwflags : u32, fpending : *mut BOOL, lpcontext : *mut *mut core::ffi::c_void) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn InitOnceComplete(lpinitonce : LPINIT_ONCE, dwflags : u32, lpcontext : *const core::ffi::c_void) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn InitializeProcThreadAttributeList(lpattributelist : *mut _PROC_THREAD_ATTRIBUTE_LIST, dwattributecount : u32, dwflags : u32, lpsize : *mut usize) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn IsThreadAFiber() -> BOOL);
windows_link::link!("kernel32.dll" "system" fn LocalFree(hmem : HLOCAL) -> HLOCAL);
windows_link::link!("kernel32.dll" "system" fn LockFileEx(hfile : HANDLE, dwflags : u32, dwreserved : u32, nnumberofbytestolocklow : u32, nnumberofbytestolockhigh : u32, lpoverlapped : *mut OVERLAPPED) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn MoveFileExW(lpexistingfilename : PCWSTR, lpnewfilename : PCWSTR, dwflags : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn MultiByteToWideChar(codepage : u32, dwflags : u32, lpmultibytestr : *const i8, cbmultibyte : i32, lpwidecharstr : PWSTR, cchwidechar : i32) -> i32);
windows_link::link!("ntdll.dll" "system" fn NtCreateFile(filehandle : *mut HANDLE, desiredaccess : ACCESS_MASK, objectattributes : *const OBJECT_ATTRIBUTES, iostatusblock : *mut IO_STATUS_BLOCK, allocationsize : *const i64, fileattributes : u32, shareaccess : u32, createdisposition : u32, createoptions : u32, eabuffer : *const core::ffi::c_void, ealength : u32) -> NTSTATUS);
windows_link::link!("ntdll.dll" "system" fn NtOpenFile(filehandle : *mut HANDLE, desiredaccess : ACCESS_MASK, objectattributes : *const OBJECT_ATTRIBUTES, iostatusblock : *mut IO_STATUS_BLOCK, shareaccess : u32, openoptions : u32) -> NTSTATUS);
windows_link::link!("ntdll.dll" "system" fn NtReadFile(filehandle : HANDLE, event : HANDLE, apcroutine : PIO_APC_ROUTINE, apccontext : *const core::ffi::c_void, iostatusblock : *mut IO_STATUS_BLOCK, buffer : *mut core::ffi::c_void, length : u32, byteoffset : *const i64, key : *const u32) -> NTSTATUS);
windows_link::link!("ntdll.dll" "system" fn NtSetInformationFile(filehandle : HANDLE, iostatusblock : *mut IO_STATUS_BLOCK, fileinformation : *const core::ffi::c_void, length : u32, fileinformationclass : FILE_INFORMATION_CLASS) -> NTSTATUS);
windows_link::link!("ntdll.dll" "system" fn NtWriteFile(filehandle : HANDLE, event : HANDLE, apcroutine : PIO_APC_ROUTINE, apccontext : *const core::ffi::c_void, iostatusblock : *mut IO_STATUS_BLOCK, buffer : *const core::ffi::c_void, length : u32, byteoffset : *const i64, key : *const u32) -> NTSTATUS);
windows_link::link!("advapi32.dll" "system" fn OpenProcessToken(processhandle : HANDLE, desiredaccess : u32, tokenhandle : *mut HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn QueryPerformanceCounter(lpperformancecount : *mut i64) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn QueryPerformanceFrequency(lpfrequency : *mut i64) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn ReadConsoleW(hconsoleinput : HANDLE, lpbuffer : *mut core::ffi::c_void, nnumberofcharstoread : u32, lpnumberofcharsread : *mut u32, pinputcontrol : *const CONSOLE_READCONSOLE_CONTROL) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn ReadFile(hfile : HANDLE, lpbuffer : *mut core::ffi::c_void, nnumberofbytestoread : u32, lpnumberofbytesread : *mut u32, lpoverlapped : *mut OVERLAPPED) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn ReadFileEx(hfile : HANDLE, lpbuffer : *mut core::ffi::c_void, nnumberofbytestoread : u32, lpoverlapped : *mut OVERLAPPED, lpcompletionroutine : LPOVERLAPPED_COMPLETION_ROUTINE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn ReleaseSRWLockExclusive(srwlock : *mut RTL_SRWLOCK));
windows_link::link!("kernel32.dll" "system" fn ReleaseSRWLockShared(srwlock : *mut RTL_SRWLOCK));
windows_link::link!("kernel32.dll" "system" fn RemoveDirectoryW(lppathname : PCWSTR) -> BOOL);
windows_link::link!("advapi32.dll" "system" "SystemFunction036" fn RtlGenRandom(randombuffer : *mut core::ffi::c_void, randombufferlength : u32) -> bool);
windows_link::link!("ntdll.dll" "system" fn RtlNtStatusToDosError(status : NTSTATUS) -> u32);
windows_link::link!("kernel32.dll" "system" fn SetCurrentDirectoryW(lppathname : PCWSTR) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetEnvironmentVariableW(lpname : PCWSTR, lpvalue : PCWSTR) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetFileAttributesW(lpfilename : PCWSTR, dwfileattributes : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetFileInformationByHandle(hfile : HANDLE, fileinformationclass : FILE_INFO_BY_HANDLE_CLASS, lpfileinformation : *const core::ffi::c_void, dwbuffersize : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetFilePointerEx(hfile : HANDLE, lidistancetomove : i64, lpnewfilepointer : *mut i64, dwmovemethod : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetFileTime(hfile : HANDLE, lpcreationtime : *const FILETIME, lplastaccesstime : *const FILETIME, lplastwritetime : *const FILETIME) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetHandleInformation(hobject : HANDLE, dwmask : u32, dwflags : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetLastError(dwerrcode : u32));
windows_link::link!("kernel32.dll" "system" fn SetStdHandle(nstdhandle : u32, hhandle : HANDLE) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetThreadStackGuarantee(stacksizeinbytes : *mut u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SetWaitableTimer(htimer : HANDLE, lpduetime : *const i64, lperiod : i32, pfncompletionroutine : PTIMERAPCROUTINE, lpargtocompletionroutine : *const core::ffi::c_void, fresume : BOOL) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn Sleep(dwmilliseconds : u32));
windows_link::link!("kernel32.dll" "system" fn SleepConditionVariableSRW(conditionvariable : *mut RTL_CONDITION_VARIABLE, srwlock : *mut RTL_SRWLOCK, dwmilliseconds : u32, flags : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn SleepEx(dwmilliseconds : u32, balertable : BOOL) -> u32);
windows_link::link!("kernel32.dll" "system" fn SwitchToThread() -> BOOL);
windows_link::link!("kernel32.dll" "system" fn TerminateProcess(hprocess : HANDLE, uexitcode : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn TlsAlloc() -> u32);
windows_link::link!("kernel32.dll" "system" fn TlsFree(dwtlsindex : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn TlsGetValue(dwtlsindex : u32) -> *mut core::ffi::c_void);
windows_link::link!("kernel32.dll" "system" fn TlsSetValue(dwtlsindex : u32, lptlsvalue : *const core::ffi::c_void) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn TryAcquireSRWLockExclusive(srwlock : *mut RTL_SRWLOCK) -> bool);
windows_link::link!("kernel32.dll" "system" fn TryAcquireSRWLockShared(srwlock : *mut RTL_SRWLOCK) -> bool);
windows_link::link!("kernel32.dll" "system" fn UnlockFile(hfile : HANDLE, dwfileoffsetlow : u32, dwfileoffsethigh : u32, nnumberofbytestounlocklow : u32, nnumberofbytestounlockhigh : u32) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn UpdateProcThreadAttribute(lpattributelist : *mut _PROC_THREAD_ATTRIBUTE_LIST, dwflags : u32, attribute : usize, lpvalue : *const core::ffi::c_void, cbsize : usize, lppreviousvalue : *mut core::ffi::c_void, lpreturnsize : *const usize) -> BOOL);
windows_link::link!("ws2_32.dll" "system" fn WSACleanup() -> i32);
windows_link::link!("ws2_32.dll" "system" fn WSADuplicateSocketW(s : SOCKET, dwprocessid : u32, lpprotocolinfo : *mut WSAPROTOCOL_INFOW) -> i32);
windows_link::link!("ws2_32.dll" "system" fn WSAGetLastError() -> i32);
windows_link::link!("ws2_32.dll" "system" fn WSARecv(s : SOCKET, lpbuffers : *const WSABUF, dwbuffercount : u32, lpnumberofbytesrecvd : *mut u32, lpflags : *mut u32, lpoverlapped : *mut OVERLAPPED, lpcompletionroutine : LPWSAOVERLAPPED_COMPLETION_ROUTINE) -> i32);
windows_link::link!("ws2_32.dll" "system" fn WSASend(s : SOCKET, lpbuffers : *const WSABUF, dwbuffercount : u32, lpnumberofbytessent : *mut u32, dwflags : u32, lpoverlapped : *mut OVERLAPPED, lpcompletionroutine : LPWSAOVERLAPPED_COMPLETION_ROUTINE) -> i32);
windows_link::link!("ws2_32.dll" "system" fn WSASocketW(af : i32, r#type : i32, protocol : i32, lpprotocolinfo : *const WSAPROTOCOL_INFOW, g : GROUP, dwflags : u32) -> SOCKET);
windows_link::link!("ws2_32.dll" "system" fn WSAStartup(wversionrequested : u16, lpwsadata : *mut WSADATA) -> i32);
windows_link::link!("kernel32.dll" "system" fn WaitForMultipleObjects(ncount : u32, lphandles : *const HANDLE, bwaitall : BOOL, dwmilliseconds : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn WaitForSingleObject(hhandle : HANDLE, dwmilliseconds : u32) -> u32);
windows_link::link!("kernel32.dll" "system" fn WakeAllConditionVariable(conditionvariable : *mut RTL_CONDITION_VARIABLE));
windows_link::link!("kernel32.dll" "system" fn WakeConditionVariable(conditionvariable : *mut RTL_CONDITION_VARIABLE));
windows_link::link!("kernel32.dll" "system" fn WideCharToMultiByte(codepage : u32, dwflags : u32, lpwidecharstr : *const u16, cchwidechar : i32, lpmultibytestr : PSTR, cbmultibyte : i32, lpdefaultchar : *const i8, lpuseddefaultchar : *mut BOOL) -> i32);
windows_link::link!("kernel32.dll" "system" fn WriteConsoleW(hconsoleoutput : HANDLE, lpbuffer : *const core::ffi::c_void, nnumberofcharstowrite : u32, lpnumberofcharswritten : *mut u32, lpreserved : *const core::ffi::c_void) -> BOOL);
windows_link::link!("kernel32.dll" "system" fn WriteFileEx(hfile : HANDLE, lpbuffer : *const core::ffi::c_void, nnumberofbytestowrite : u32, lpoverlapped : *mut OVERLAPPED, lpcompletionroutine : LPOVERLAPPED_COMPLETION_ROUTINE) -> BOOL);
windows_link::link!("ws2_32.dll" "system" fn accept(s : SOCKET, addr : *mut SOCKADDR, addrlen : *mut i32) -> SOCKET);
windows_link::link!("ws2_32.dll" "system" fn bind(s : SOCKET, name : *const SOCKADDR, namelen : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn closesocket(s : SOCKET) -> i32);
windows_link::link!("ws2_32.dll" "system" fn connect(s : SOCKET, name : *const SOCKADDR, namelen : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn freeaddrinfo(paddrinfo : *const ADDRINFOA));
windows_link::link!("ws2_32.dll" "system" fn getaddrinfo(pnodename : PCSTR, pservicename : PCSTR, phints : *const ADDRINFOA, ppresult : *mut PADDRINFOA) -> i32);
windows_link::link!("ws2_32.dll" "system" fn getpeername(s : SOCKET, name : *mut SOCKADDR, namelen : *mut i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn getsockname(s : SOCKET, name : *mut SOCKADDR, namelen : *mut i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn getsockopt(s : SOCKET, level : i32, optname : i32, optval : *mut i8, optlen : *mut i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn ioctlsocket(s : SOCKET, cmd : i32, argp : *mut u_long) -> i32);
windows_link::link!("ws2_32.dll" "system" fn listen(s : SOCKET, backlog : i32) -> i32);
windows_link::link!("kernel32.dll" "system" fn lstrlenW(lpstring : PCWSTR) -> i32);
windows_link::link!("ws2_32.dll" "system" fn recv(s : SOCKET, buf : *mut i8, len : i32, flags : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn recvfrom(s : SOCKET, buf : *mut i8, len : i32, flags : i32, from : *mut SOCKADDR, fromlen : *mut i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn select(nfds : i32, readfds : *mut fd_set, writefds : *mut fd_set, exceptfds : *mut fd_set, timeout : *const timeval) -> i32);
windows_link::link!("ws2_32.dll" "system" fn send(s : SOCKET, buf : *const i8, len : i32, flags : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn sendto(s : SOCKET, buf : *const i8, len : i32, flags : i32, to : *const SOCKADDR, tolen : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn setsockopt(s : SOCKET, level : i32, optname : i32, optval : *const i8, optlen : i32) -> i32);
windows_link::link!("ws2_32.dll" "system" fn shutdown(s : SOCKET, how : i32) -> i32);
pub const ABOVE_NORMAL_PRIORITY_CLASS: i32 = 32768;
pub type ACCESS_MASK = u32;
pub type ADDRESS_FAMILY = u16;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ADDRINFOA {
    pub ai_flags: i32,
    pub ai_family: i32,
    pub ai_socktype: i32,
    pub ai_protocol: i32,
    pub ai_addrlen: usize,
    pub ai_canonname: *mut i8,
    pub ai_addr: *mut SOCKADDR,
    pub ai_next: *mut Self,
}
impl Default for ADDRINFOA {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const AF_INET: i32 = 2;
pub const AF_INET6: i32 = 23;
pub const AF_UNIX: i32 = 1;
pub const AF_UNSPEC: i32 = 0;
pub const ALL_PROCESSOR_GROUPS: i32 = 65535;
#[repr(C, align(16))]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct ARM64_NT_CONTEXT {
    pub ContextFlags: u32,
    pub Cpsr: u32,
    pub Anonymous: ARM64_NT_CONTEXT_0,
    pub Sp: u64,
    pub Pc: u64,
    pub V: [ARM64_NT_NEON128; 32],
    pub Fpcr: u32,
    pub Fpsr: u32,
    pub Bcr: [u32; 8],
    pub Bvr: [u64; 8],
    pub Wcr: [u32; 2],
    pub Wvr: [u64; 2],
}
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
impl Default for ARM64_NT_CONTEXT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub union ARM64_NT_CONTEXT_0 {
    pub Anonymous: ARM64_NT_CONTEXT_0_0,
    pub X: [u64; 31],
}
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
impl Default for ARM64_NT_CONTEXT_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
#[derive(Clone, Copy, Default)]
pub struct ARM64_NT_CONTEXT_0_0 {
    pub X0: u64,
    pub X1: u64,
    pub X2: u64,
    pub X3: u64,
    pub X4: u64,
    pub X5: u64,
    pub X6: u64,
    pub X7: u64,
    pub X8: u64,
    pub X9: u64,
    pub X10: u64,
    pub X11: u64,
    pub X12: u64,
    pub X13: u64,
    pub X14: u64,
    pub X15: u64,
    pub X16: u64,
    pub X17: u64,
    pub X18: u64,
    pub X19: u64,
    pub X20: u64,
    pub X21: u64,
    pub X22: u64,
    pub X23: u64,
    pub X24: u64,
    pub X25: u64,
    pub X26: u64,
    pub X27: u64,
    pub X28: u64,
    pub Fp: u64,
    pub Lr: u64,
}
#[repr(C, align(16))]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub struct ARM64_NT_CONTEXT {
    pub ContextFlags: u32,
    pub Cpsr: u32,
    pub Anonymous: ARM64_NT_CONTEXT_0,
    pub Sp: u64,
    pub Pc: u64,
    pub V: [NEON128; 32],
    pub Fpcr: u32,
    pub Fpsr: u32,
    pub Bcr: [u32; 8],
    pub Bvr: [u64; 8],
    pub Wcr: [u32; 2],
    pub Wvr: [u64; 2],
}
#[cfg(target_arch = "aarch64")]
impl Default for ARM64_NT_CONTEXT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
pub union ARM64_NT_CONTEXT_0 {
    pub Anonymous: ARM64_NT_CONTEXT_0_0,
    pub X: [u64; 31],
}
#[cfg(target_arch = "aarch64")]
impl Default for ARM64_NT_CONTEXT_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Default)]
pub struct ARM64_NT_CONTEXT_0_0 {
    pub X0: u64,
    pub X1: u64,
    pub X2: u64,
    pub X3: u64,
    pub X4: u64,
    pub X5: u64,
    pub X6: u64,
    pub X7: u64,
    pub X8: u64,
    pub X9: u64,
    pub X10: u64,
    pub X11: u64,
    pub X12: u64,
    pub X13: u64,
    pub X14: u64,
    pub X15: u64,
    pub X16: u64,
    pub X17: u64,
    pub X18: u64,
    pub X19: u64,
    pub X20: u64,
    pub X21: u64,
    pub X22: u64,
    pub X23: u64,
    pub X24: u64,
    pub X25: u64,
    pub X26: u64,
    pub X27: u64,
    pub X28: u64,
    pub Fp: u64,
    pub Lr: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union ARM64_NT_NEON128 {
    pub Anonymous: ARM64_NT_NEON128_0,
    pub D: [f64; 2],
    pub S: [f32; 4],
    pub H: [u16; 8],
    pub B: [u8; 16],
}
impl Default for ARM64_NT_NEON128 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ARM64_NT_NEON128_0 {
    pub Low: u64,
    pub High: i64,
}
pub const BELOW_NORMAL_PRIORITY_CLASS: i32 = 16384;
pub type BOOL = i32;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct BY_HANDLE_FILE_INFORMATION {
    pub dwFileAttributes: u32,
    pub ftCreationTime: FILETIME,
    pub ftLastAccessTime: FILETIME,
    pub ftLastWriteTime: FILETIME,
    pub dwVolumeSerialNumber: u32,
    pub nFileSizeHigh: u32,
    pub nFileSizeLow: u32,
    pub nNumberOfLinks: u32,
    pub nFileIndexHigh: u32,
    pub nFileIndexLow: u32,
}
pub const CALLBACK_CHUNK_FINISHED: i32 = 0;
pub const CALLBACK_STREAM_SWITCH: i32 = 1;
pub type CCHAR = i8;
pub type CONDITION_VARIABLE = RTL_CONDITION_VARIABLE;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct CONSOLE_READCONSOLE_CONTROL {
    pub nLength: u32,
    pub nInitialChars: u32,
    pub dwCtrlWakeupMask: u32,
    pub dwControlKeyState: u32,
}
#[repr(C)]
#[cfg(target_arch = "x86")]
#[derive(Clone, Copy)]
pub struct CONTEXT {
    pub ContextFlags: u32,
    pub Dr0: u32,
    pub Dr1: u32,
    pub Dr2: u32,
    pub Dr3: u32,
    pub Dr6: u32,
    pub Dr7: u32,
    pub FloatSave: FLOATING_SAVE_AREA,
    pub SegGs: u32,
    pub SegFs: u32,
    pub SegEs: u32,
    pub SegDs: u32,
    pub Edi: u32,
    pub Esi: u32,
    pub Ebx: u32,
    pub Edx: u32,
    pub Ecx: u32,
    pub Eax: u32,
    pub Ebp: u32,
    pub Eip: u32,
    pub SegCs: u32,
    pub EFlags: u32,
    pub Esp: u32,
    pub SegSs: u32,
    pub ExtendedRegisters: [u8; 512],
}
#[cfg(target_arch = "x86")]
impl Default for CONTEXT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct CONTEXT {
    pub P1Home: u64,
    pub P2Home: u64,
    pub P3Home: u64,
    pub P4Home: u64,
    pub P5Home: u64,
    pub P6Home: u64,
    pub ContextFlags: u32,
    pub MxCsr: u32,
    pub SegCs: u16,
    pub SegDs: u16,
    pub SegEs: u16,
    pub SegFs: u16,
    pub SegGs: u16,
    pub SegSs: u16,
    pub EFlags: u32,
    pub Dr0: u64,
    pub Dr1: u64,
    pub Dr2: u64,
    pub Dr3: u64,
    pub Dr6: u64,
    pub Dr7: u64,
    pub Rax: u64,
    pub Rcx: u64,
    pub Rdx: u64,
    pub Rbx: u64,
    pub Rsp: u64,
    pub Rbp: u64,
    pub Rsi: u64,
    pub Rdi: u64,
    pub R8: u64,
    pub R9: u64,
    pub R10: u64,
    pub R11: u64,
    pub R12: u64,
    pub R13: u64,
    pub R14: u64,
    pub R15: u64,
    pub Rip: u64,
    pub Anonymous: CONTEXT_0,
    pub VectorRegister: [M128A; 26],
    pub VectorControl: u64,
    pub DebugControl: u64,
    pub LastBranchToRip: u64,
    pub LastBranchFromRip: u64,
    pub LastExceptionToRip: u64,
    pub LastExceptionFromRip: u64,
}
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
impl Default for CONTEXT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub union CONTEXT_0 {
    pub FltSave: XMM_SAVE_AREA32,
    pub Anonymous: CONTEXT_0_0,
}
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
impl Default for CONTEXT_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct CONTEXT_0_0 {
    pub Header: [M128A; 2],
    pub Legacy: [M128A; 8],
    pub Xmm0: M128A,
    pub Xmm1: M128A,
    pub Xmm2: M128A,
    pub Xmm3: M128A,
    pub Xmm4: M128A,
    pub Xmm5: M128A,
    pub Xmm6: M128A,
    pub Xmm7: M128A,
    pub Xmm8: M128A,
    pub Xmm9: M128A,
    pub Xmm10: M128A,
    pub Xmm11: M128A,
    pub Xmm12: M128A,
    pub Xmm13: M128A,
    pub Xmm14: M128A,
    pub Xmm15: M128A,
}
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
impl Default for CONTEXT_0_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[cfg(target_arch = "aarch64")]
pub type CONTEXT = ARM64_NT_CONTEXT;
pub const CP_UTF8: i32 = 65001;
pub const CREATE_ALWAYS: i32 = 2;
pub const CREATE_BREAKAWAY_FROM_JOB: i32 = 16777216;
pub const CREATE_DEFAULT_ERROR_MODE: i32 = 67108864;
pub const CREATE_FORCEDOS: i32 = 8192;
pub const CREATE_IGNORE_SYSTEM_DEFAULT: u32 = 2147483648;
pub const CREATE_NEW: i32 = 1;
pub const CREATE_NEW_CONSOLE: i32 = 16;
pub const CREATE_NEW_PROCESS_GROUP: i32 = 512;
pub const CREATE_NO_WINDOW: i32 = 134217728;
pub const CREATE_PRESERVE_CODE_AUTHZ_LEVEL: i32 = 33554432;
pub const CREATE_PROTECTED_PROCESS: i32 = 262144;
pub const CREATE_SECURE_PROCESS: i32 = 4194304;
pub const CREATE_SEPARATE_WOW_VDM: i32 = 2048;
pub const CREATE_SHARED_WOW_VDM: i32 = 4096;
pub const CREATE_SUSPENDED: i32 = 4;
pub const CREATE_UNICODE_ENVIRONMENT: i32 = 1024;
pub const CREATE_WAITABLE_TIMER_HIGH_RESOLUTION: i32 = 2;
pub const CREATE_WAITABLE_TIMER_MANUAL_RESET: i32 = 1;
pub const CSTR_EQUAL: i32 = 2;
pub const CSTR_GREATER_THAN: i32 = 3;
pub const CSTR_LESS_THAN: i32 = 1;
pub const DEBUG_ONLY_THIS_PROCESS: i32 = 2;
pub const DEBUG_PROCESS: i32 = 1;
pub const DELETE: i32 = 65536;
pub const DETACHED_PROCESS: i32 = 8;
pub const DISABLE_NEWLINE_AUTO_RETURN: i32 = 8;
pub const DLL_PROCESS_DETACH: i32 = 0;
pub const DLL_THREAD_DETACH: i32 = 3;
pub const DNS_ERROR_ADDRESS_REQUIRED: i32 = 9573;
pub const DNS_ERROR_ALIAS_LOOP: i32 = 9722;
pub const DNS_ERROR_AUTOZONE_ALREADY_EXISTS: i32 = 9610;
pub const DNS_ERROR_AXFR: i32 = 9752;
pub const DNS_ERROR_BACKGROUND_LOADING: i32 = 9568;
pub const DNS_ERROR_BAD_KEYMASTER: i32 = 9122;
pub const DNS_ERROR_BAD_PACKET: i32 = 9502;
pub const DNS_ERROR_CANNOT_FIND_ROOT_HINTS: i32 = 9564;
pub const DNS_ERROR_CLIENT_SUBNET_ALREADY_EXISTS: i32 = 9977;
pub const DNS_ERROR_CLIENT_SUBNET_DOES_NOT_EXIST: i32 = 9976;
pub const DNS_ERROR_CLIENT_SUBNET_IS_ACCESSED: i32 = 9975;
pub const DNS_ERROR_CNAME_COLLISION: i32 = 9709;
pub const DNS_ERROR_CNAME_LOOP: i32 = 9707;
pub const DNS_ERROR_DATAFILE_OPEN_FAILURE: i32 = 9653;
pub const DNS_ERROR_DATAFILE_PARSING: i32 = 9655;
pub const DNS_ERROR_DEFAULT_SCOPE: i32 = 9960;
pub const DNS_ERROR_DEFAULT_VIRTUALIZATION_INSTANCE: i32 = 9925;
pub const DNS_ERROR_DEFAULT_ZONESCOPE: i32 = 9953;
pub const DNS_ERROR_DELEGATION_REQUIRED: i32 = 9571;
pub const DNS_ERROR_DNAME_COLLISION: i32 = 9721;
pub const DNS_ERROR_DNSSEC_IS_DISABLED: i32 = 9125;
pub const DNS_ERROR_DP_ALREADY_ENLISTED: i32 = 9904;
pub const DNS_ERROR_DP_ALREADY_EXISTS: i32 = 9902;
pub const DNS_ERROR_DP_DOES_NOT_EXIST: i32 = 9901;
pub const DNS_ERROR_DP_FSMO_ERROR: i32 = 9906;
pub const DNS_ERROR_DP_NOT_AVAILABLE: i32 = 9905;
pub const DNS_ERROR_DP_NOT_ENLISTED: i32 = 9903;
pub const DNS_ERROR_DS_UNAVAILABLE: i32 = 9717;
pub const DNS_ERROR_DS_ZONE_ALREADY_EXISTS: i32 = 9718;
pub const DNS_ERROR_DWORD_VALUE_TOO_LARGE: i32 = 9567;
pub const DNS_ERROR_DWORD_VALUE_TOO_SMALL: i32 = 9566;
pub const DNS_ERROR_FILE_WRITEBACK_FAILED: i32 = 9654;
pub const DNS_ERROR_FORWARDER_ALREADY_EXISTS: i32 = 9619;
pub const DNS_ERROR_INCONSISTENT_ROOT_HINTS: i32 = 9565;
pub const DNS_ERROR_INVAILD_VIRTUALIZATION_INSTANCE_NAME: i32 = 9924;
pub const DNS_ERROR_INVALID_CLIENT_SUBNET_NAME: i32 = 9984;
pub const DNS_ERROR_INVALID_DATA: i32 = 13;
pub const DNS_ERROR_INVALID_DATAFILE_NAME: i32 = 9652;
pub const DNS_ERROR_INVALID_INITIAL_ROLLOVER_OFFSET: i32 = 9115;
pub const DNS_ERROR_INVALID_IP_ADDRESS: i32 = 9552;
pub const DNS_ERROR_INVALID_KEY_SIZE: i32 = 9106;
pub const DNS_ERROR_INVALID_NAME: i32 = 123;
pub const DNS_ERROR_INVALID_NAME_CHAR: i32 = 9560;
pub const DNS_ERROR_INVALID_NSEC3_ITERATION_COUNT: i32 = 9124;
pub const DNS_ERROR_INVALID_POLICY_TABLE: i32 = 9572;
pub const DNS_ERROR_INVALID_PROPERTY: i32 = 9553;
pub const DNS_ERROR_INVALID_ROLLOVER_PERIOD: i32 = 9114;
pub const DNS_ERROR_INVALID_SCOPE_NAME: i32 = 9958;
pub const DNS_ERROR_INVALID_SCOPE_OPERATION: i32 = 9961;
pub const DNS_ERROR_INVALID_SIGNATURE_VALIDITY_PERIOD: i32 = 9123;
pub const DNS_ERROR_INVALID_TYPE: i32 = 9551;
pub const DNS_ERROR_INVALID_XML: i32 = 9126;
pub const DNS_ERROR_INVALID_ZONESCOPE_NAME: i32 = 9954;
pub const DNS_ERROR_INVALID_ZONE_OPERATION: i32 = 9603;
pub const DNS_ERROR_INVALID_ZONE_TYPE: i32 = 9611;
pub const DNS_ERROR_KEYMASTER_REQUIRED: i32 = 9101;
pub const DNS_ERROR_KSP_DOES_NOT_SUPPORT_PROTECTION: i32 = 9108;
pub const DNS_ERROR_KSP_NOT_ACCESSIBLE: i32 = 9112;
pub const DNS_ERROR_LOAD_ZONESCOPE_FAILED: i32 = 9956;
pub const DNS_ERROR_NAME_DOES_NOT_EXIST: i32 = 9714;
pub const DNS_ERROR_NAME_NOT_IN_ZONE: i32 = 9706;
pub const DNS_ERROR_NBSTAT_INIT_FAILED: i32 = 9617;
pub const DNS_ERROR_NEED_SECONDARY_ADDRESSES: i32 = 9614;
pub const DNS_ERROR_NEED_WINS_SERVERS: i32 = 9616;
pub const DNS_ERROR_NODE_CREATION_FAILED: i32 = 9703;
pub const DNS_ERROR_NODE_IS_CNAME: i32 = 9708;
pub const DNS_ERROR_NODE_IS_DNAME: i32 = 9720;
pub const DNS_ERROR_NON_RFC_NAME: i32 = 9556;
pub const DNS_ERROR_NOT_ALLOWED_ON_ACTIVE_SKD: i32 = 9119;
pub const DNS_ERROR_NOT_ALLOWED_ON_RODC: i32 = 9569;
pub const DNS_ERROR_NOT_ALLOWED_ON_ROOT_SERVER: i32 = 9562;
pub const DNS_ERROR_NOT_ALLOWED_ON_SIGNED_ZONE: i32 = 9102;
pub const DNS_ERROR_NOT_ALLOWED_ON_UNSIGNED_ZONE: i32 = 9121;
pub const DNS_ERROR_NOT_ALLOWED_ON_ZSK: i32 = 9118;
pub const DNS_ERROR_NOT_ALLOWED_UNDER_DELEGATION: i32 = 9563;
pub const DNS_ERROR_NOT_ALLOWED_UNDER_DNAME: i32 = 9570;
pub const DNS_ERROR_NOT_ALLOWED_WITH_ZONESCOPES: i32 = 9955;
pub const DNS_ERROR_NOT_ENOUGH_SIGNING_KEY_DESCRIPTORS: i32 = 9104;
pub const DNS_ERROR_NOT_UNIQUE: i32 = 9555;
pub const DNS_ERROR_NO_BOOTFILE_IF_DS_ZONE: i32 = 9719;
pub const DNS_ERROR_NO_CREATE_CACHE_DATA: i32 = 9713;
pub const DNS_ERROR_NO_DNS_SERVERS: i32 = 9852;
pub const DNS_ERROR_NO_MEMORY: i32 = 14;
pub const DNS_ERROR_NO_PACKET: i32 = 9503;
pub const DNS_ERROR_NO_TCPIP: i32 = 9851;
pub const DNS_ERROR_NO_VALID_TRUST_ANCHORS: i32 = 9127;
pub const DNS_ERROR_NO_ZONE_INFO: i32 = 9602;
pub const DNS_ERROR_NSEC3_INCOMPATIBLE_WITH_RSA_SHA1: i32 = 9103;
pub const DNS_ERROR_NSEC3_NAME_COLLISION: i32 = 9129;
pub const DNS_ERROR_NSEC_INCOMPATIBLE_WITH_NSEC3_RSA_SHA1: i32 = 9130;
pub const DNS_ERROR_NUMERIC_NAME: i32 = 9561;
pub const DNS_ERROR_POLICY_ALREADY_EXISTS: i32 = 9971;
pub const DNS_ERROR_POLICY_DOES_NOT_EXIST: i32 = 9972;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA: i32 = 9973;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_CLIENT_SUBNET: i32 = 9990;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_FQDN: i32 = 9994;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_INTERFACE: i32 = 9993;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_NETWORK_PROTOCOL: i32 = 9992;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_QUERY_TYPE: i32 = 9995;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_TIME_OF_DAY: i32 = 9996;
pub const DNS_ERROR_POLICY_INVALID_CRITERIA_TRANSPORT_PROTOCOL: i32 = 9991;
pub const DNS_ERROR_POLICY_INVALID_NAME: i32 = 9982;
pub const DNS_ERROR_POLICY_INVALID_SETTINGS: i32 = 9974;
pub const DNS_ERROR_POLICY_INVALID_WEIGHT: i32 = 9981;
pub const DNS_ERROR_POLICY_LOCKED: i32 = 9980;
pub const DNS_ERROR_POLICY_MISSING_CRITERIA: i32 = 9983;
pub const DNS_ERROR_POLICY_PROCESSING_ORDER_INVALID: i32 = 9985;
pub const DNS_ERROR_POLICY_SCOPE_MISSING: i32 = 9986;
pub const DNS_ERROR_POLICY_SCOPE_NOT_ALLOWED: i32 = 9987;
pub const DNS_ERROR_PRIMARY_REQUIRES_DATAFILE: i32 = 9651;
pub const DNS_ERROR_RCODE: i32 = 9504;
pub const DNS_ERROR_RCODE_BADKEY: i32 = 9017;
pub const DNS_ERROR_RCODE_BADSIG: i32 = 9016;
pub const DNS_ERROR_RCODE_BADTIME: i32 = 9018;
pub const DNS_ERROR_RCODE_FORMAT_ERROR: i32 = 9001;
pub const DNS_ERROR_RCODE_NAME_ERROR: i32 = 9003;
pub const DNS_ERROR_RCODE_NOTAUTH: i32 = 9009;
pub const DNS_ERROR_RCODE_NOTZONE: i32 = 9010;
pub const DNS_ERROR_RCODE_NOT_IMPLEMENTED: i32 = 9004;
pub const DNS_ERROR_RCODE_NXRRSET: i32 = 9008;
pub const DNS_ERROR_RCODE_REFUSED: i32 = 9005;
pub const DNS_ERROR_RCODE_SERVER_FAILURE: i32 = 9002;
pub const DNS_ERROR_RCODE_YXDOMAIN: i32 = 9006;
pub const DNS_ERROR_RCODE_YXRRSET: i32 = 9007;
pub const DNS_ERROR_RECORD_ALREADY_EXISTS: i32 = 9711;
pub const DNS_ERROR_RECORD_DOES_NOT_EXIST: i32 = 9701;
pub const DNS_ERROR_RECORD_FORMAT: i32 = 9702;
pub const DNS_ERROR_RECORD_ONLY_AT_ZONE_ROOT: i32 = 9710;
pub const DNS_ERROR_RECORD_TIMED_OUT: i32 = 9705;
pub const DNS_ERROR_ROLLOVER_ALREADY_QUEUED: i32 = 9120;
pub const DNS_ERROR_ROLLOVER_IN_PROGRESS: i32 = 9116;
pub const DNS_ERROR_ROLLOVER_NOT_POKEABLE: i32 = 9128;
pub const DNS_ERROR_RRL_INVALID_IPV4_PREFIX: i32 = 9913;
pub const DNS_ERROR_RRL_INVALID_IPV6_PREFIX: i32 = 9914;
pub const DNS_ERROR_RRL_INVALID_LEAK_RATE: i32 = 9916;
pub const DNS_ERROR_RRL_INVALID_TC_RATE: i32 = 9915;
pub const DNS_ERROR_RRL_INVALID_WINDOW_SIZE: i32 = 9912;
pub const DNS_ERROR_RRL_LEAK_RATE_LESSTHAN_TC_RATE: i32 = 9917;
pub const DNS_ERROR_RRL_NOT_ENABLED: i32 = 9911;
pub const DNS_ERROR_SCOPE_ALREADY_EXISTS: i32 = 9963;
pub const DNS_ERROR_SCOPE_DOES_NOT_EXIST: i32 = 9959;
pub const DNS_ERROR_SCOPE_LOCKED: i32 = 9962;
pub const DNS_ERROR_SECONDARY_DATA: i32 = 9712;
pub const DNS_ERROR_SECONDARY_REQUIRES_MASTER_IP: i32 = 9612;
pub const DNS_ERROR_SERVERSCOPE_IS_REFERENCED: i32 = 9988;
pub const DNS_ERROR_SIGNING_KEY_NOT_ACCESSIBLE: i32 = 9107;
pub const DNS_ERROR_SOA_DELETE_INVALID: i32 = 9618;
pub const DNS_ERROR_STANDBY_KEY_NOT_PRESENT: i32 = 9117;
pub const DNS_ERROR_SUBNET_ALREADY_EXISTS: i32 = 9979;
pub const DNS_ERROR_SUBNET_DOES_NOT_EXIST: i32 = 9978;
pub const DNS_ERROR_TOO_MANY_SKDS: i32 = 9113;
pub const DNS_ERROR_TRY_AGAIN_LATER: i32 = 9554;
pub const DNS_ERROR_UNEXPECTED_CNG_ERROR: i32 = 9110;
pub const DNS_ERROR_UNEXPECTED_DATA_PROTECTION_ERROR: i32 = 9109;
pub const DNS_ERROR_UNKNOWN_RECORD_TYPE: i32 = 9704;
pub const DNS_ERROR_UNKNOWN_SIGNING_PARAMETER_VERSION: i32 = 9111;
pub const DNS_ERROR_UNSECURE_PACKET: i32 = 9505;
pub const DNS_ERROR_UNSUPPORTED_ALGORITHM: i32 = 9105;
pub const DNS_ERROR_VIRTUALIZATION_INSTANCE_ALREADY_EXISTS: i32 = 9921;
pub const DNS_ERROR_VIRTUALIZATION_INSTANCE_DOES_NOT_EXIST: i32 = 9922;
pub const DNS_ERROR_VIRTUALIZATION_TREE_LOCKED: i32 = 9923;
pub const DNS_ERROR_WINS_INIT_FAILED: i32 = 9615;
pub const DNS_ERROR_ZONESCOPE_ALREADY_EXISTS: i32 = 9951;
pub const DNS_ERROR_ZONESCOPE_DOES_NOT_EXIST: i32 = 9952;
pub const DNS_ERROR_ZONESCOPE_FILE_WRITEBACK_FAILED: i32 = 9957;
pub const DNS_ERROR_ZONESCOPE_IS_REFERENCED: i32 = 9989;
pub const DNS_ERROR_ZONE_ALREADY_EXISTS: i32 = 9609;
pub const DNS_ERROR_ZONE_CONFIGURATION_ERROR: i32 = 9604;
pub const DNS_ERROR_ZONE_CREATION_FAILED: i32 = 9608;
pub const DNS_ERROR_ZONE_DOES_NOT_EXIST: i32 = 9601;
pub const DNS_ERROR_ZONE_HAS_NO_NS_RECORDS: i32 = 9606;
pub const DNS_ERROR_ZONE_HAS_NO_SOA_RECORD: i32 = 9605;
pub const DNS_ERROR_ZONE_IS_SHUTDOWN: i32 = 9621;
pub const DNS_ERROR_ZONE_LOCKED: i32 = 9607;
pub const DNS_ERROR_ZONE_LOCKED_FOR_SIGNING: i32 = 9622;
pub const DNS_ERROR_ZONE_NOT_SECONDARY: i32 = 9613;
pub const DNS_ERROR_ZONE_REQUIRES_MASTER_IP: i32 = 9620;
pub const DUPLICATE_CLOSE_SOURCE: i32 = 1;
pub const DUPLICATE_SAME_ACCESS: i32 = 2;
pub const ENABLE_AUTO_POSITION: i32 = 256;
pub const ENABLE_ECHO_INPUT: i32 = 4;
pub const ENABLE_EXTENDED_FLAGS: i32 = 128;
pub const ENABLE_INSERT_MODE: i32 = 32;
pub const ENABLE_LINE_INPUT: i32 = 2;
pub const ENABLE_LVB_GRID_WORLDWIDE: i32 = 16;
pub const ENABLE_MOUSE_INPUT: i32 = 16;
pub const ENABLE_PROCESSED_INPUT: i32 = 1;
pub const ENABLE_PROCESSED_OUTPUT: i32 = 1;
pub const ENABLE_QUICK_EDIT_MODE: i32 = 64;
pub const ENABLE_VIRTUAL_TERMINAL_INPUT: i32 = 512;
pub const ENABLE_VIRTUAL_TERMINAL_PROCESSING: i32 = 4;
pub const ENABLE_WINDOW_INPUT: i32 = 8;
pub const ENABLE_WRAP_AT_EOL_OUTPUT: i32 = 2;
pub const ERROR_ABANDONED_WAIT_0: i32 = 735;
pub const ERROR_ABANDONED_WAIT_63: i32 = 736;
pub const ERROR_ABANDON_HIBERFILE: i32 = 787;
pub const ERROR_ABIOS_ERROR: i32 = 538;
pub const ERROR_ACCESS_AUDIT_BY_POLICY: i32 = 785;
pub const ERROR_ACCESS_DENIED: i32 = 5;
pub const ERROR_ACCESS_DENIED_APPDATA: i32 = 502;
pub const ERROR_ACCESS_DISABLED_BY_POLICY: i32 = 1260;
pub const ERROR_ACCESS_DISABLED_NO_SAFER_UI_BY_POLICY: i32 = 786;
pub const ERROR_ACCESS_DISABLED_WEBBLADE: i32 = 1277;
pub const ERROR_ACCESS_DISABLED_WEBBLADE_TAMPER: i32 = 1278;
pub const ERROR_ACCOUNT_DISABLED: i32 = 1331;
pub const ERROR_ACCOUNT_EXPIRED: i32 = 1793;
pub const ERROR_ACCOUNT_LOCKED_OUT: i32 = 1909;
pub const ERROR_ACCOUNT_RESTRICTION: i32 = 1327;
pub const ERROR_ACPI_ERROR: i32 = 669;
pub const ERROR_ACTIVE_CONNECTIONS: i32 = 2402;
pub const ERROR_ADAP_HDW_ERR: i32 = 57;
pub const ERROR_ADDRESS_ALREADY_ASSOCIATED: i32 = 1227;
pub const ERROR_ADDRESS_NOT_ASSOCIATED: i32 = 1228;
pub const ERROR_ALERTED: i32 = 739;
pub const ERROR_ALIAS_EXISTS: i32 = 1379;
pub const ERROR_ALLOCATE_BUCKET: i32 = 602;
pub const ERROR_ALLOTTED_SPACE_EXCEEDED: i32 = 1344;
pub const ERROR_ALL_USER_TRUST_QUOTA_EXCEEDED: i32 = 1933;
pub const ERROR_ALREADY_ASSIGNED: i32 = 85;
pub const ERROR_ALREADY_EXISTS: i32 = 183;
pub const ERROR_ALREADY_FIBER: i32 = 1280;
pub const ERROR_ALREADY_HAS_STREAM_ID: i32 = 4444;
pub const ERROR_ALREADY_INITIALIZED: i32 = 1247;
pub const ERROR_ALREADY_REGISTERED: i32 = 1242;
pub const ERROR_ALREADY_RUNNING_LKG: i32 = 1074;
pub const ERROR_ALREADY_THREAD: i32 = 1281;
pub const ERROR_ALREADY_WAITING: i32 = 1904;
pub const ERROR_ALREADY_WIN32: i32 = 719;
pub const ERROR_API_UNAVAILABLE: i32 = 15841;
pub const ERROR_APPCONTAINER_REQUIRED: i32 = 4251;
pub const ERROR_APPEXEC_APP_COMPAT_BLOCK: i32 = 3068;
pub const ERROR_APPEXEC_CALLER_WAIT_TIMEOUT: i32 = 3069;
pub const ERROR_APPEXEC_CALLER_WAIT_TIMEOUT_LICENSING: i32 = 3071;
pub const ERROR_APPEXEC_CALLER_WAIT_TIMEOUT_RESOURCES: i32 = 3072;
pub const ERROR_APPEXEC_CALLER_WAIT_TIMEOUT_TERMINATION: i32 = 3070;
pub const ERROR_APPEXEC_CONDITION_NOT_SATISFIED: i32 = 3060;
pub const ERROR_APPEXEC_HANDLE_INVALIDATED: i32 = 3061;
pub const ERROR_APPEXEC_HOST_ID_MISMATCH: i32 = 3066;
pub const ERROR_APPEXEC_INVALID_HOST_GENERATION: i32 = 3062;
pub const ERROR_APPEXEC_INVALID_HOST_STATE: i32 = 3064;
pub const ERROR_APPEXEC_NO_DONOR: i32 = 3065;
pub const ERROR_APPEXEC_UNEXPECTED_PROCESS_REGISTRATION: i32 = 3063;
pub const ERROR_APPEXEC_UNKNOWN_USER: i32 = 3067;
pub const ERROR_APPHELP_BLOCK: i32 = 1259;
pub const ERROR_APPX_FILE_NOT_ENCRYPTED: i32 = 409;
pub const ERROR_APP_HANG: i32 = 1298;
pub const ERROR_APP_INIT_FAILURE: i32 = 575;
pub const ERROR_APP_WRONG_OS: i32 = 1151;
pub const ERROR_ARBITRATION_UNHANDLED: i32 = 723;
pub const ERROR_ARENA_TRASHED: i32 = 7;
pub const ERROR_ARITHMETIC_OVERFLOW: i32 = 534;
pub const ERROR_ASSERTION_FAILURE: i32 = 668;
pub const ERROR_ATOMIC_LOCKS_NOT_SUPPORTED: i32 = 174;
pub const ERROR_AUDIT_FAILED: i32 = 606;
pub const ERROR_AUTHENTICATION_FIREWALL_FAILED: i32 = 1935;
pub const ERROR_AUTHIP_FAILURE: i32 = 1469;
pub const ERROR_AUTODATASEG_EXCEEDS_64k: i32 = 199;
pub const ERROR_BACKUP_CONTROLLER: i32 = 586;
pub const ERROR_BADDB: i32 = 1009;
pub const ERROR_BADKEY: i32 = 1010;
pub const ERROR_BADSTARTPOSITION: i32 = 778;
pub const ERROR_BAD_ACCESSOR_FLAGS: i32 = 773;
pub const ERROR_BAD_ARGUMENTS: i32 = 160;
pub const ERROR_BAD_COMMAND: i32 = 22;
pub const ERROR_BAD_COMPRESSION_BUFFER: i32 = 605;
pub const ERROR_BAD_CONFIGURATION: i32 = 1610;
pub const ERROR_BAD_CURRENT_DIRECTORY: i32 = 703;
pub const ERROR_BAD_DESCRIPTOR_FORMAT: i32 = 1361;
pub const ERROR_BAD_DEVICE: i32 = 1200;
pub const ERROR_BAD_DEVICE_PATH: i32 = 330;
pub const ERROR_BAD_DEV_TYPE: i32 = 66;
pub const ERROR_BAD_DLL_ENTRYPOINT: i32 = 609;
pub const ERROR_BAD_DRIVER_LEVEL: i32 = 119;
pub const ERROR_BAD_ENVIRONMENT: i32 = 10;
pub const ERROR_BAD_EXE_FORMAT: i32 = 193;
pub const ERROR_BAD_FILE_TYPE: i32 = 222;
pub const ERROR_BAD_FORMAT: i32 = 11;
pub const ERROR_BAD_FUNCTION_TABLE: i32 = 559;
pub const ERROR_BAD_IMPERSONATION_LEVEL: i32 = 1346;
pub const ERROR_BAD_INHERITANCE_ACL: i32 = 1340;
pub const ERROR_BAD_LENGTH: i32 = 24;
pub const ERROR_BAD_LOGON_SESSION_STATE: i32 = 1365;
pub const ERROR_BAD_MCFG_TABLE: i32 = 791;
pub const ERROR_BAD_NETPATH: i32 = 53;
pub const ERROR_BAD_NET_NAME: i32 = 67;
pub const ERROR_BAD_NET_RESP: i32 = 58;
pub const ERROR_BAD_PATHNAME: i32 = 161;
pub const ERROR_BAD_PIPE: i32 = 230;
pub const ERROR_BAD_PROFILE: i32 = 1206;
pub const ERROR_BAD_PROVIDER: i32 = 1204;
pub const ERROR_BAD_QUERY_SYNTAX: i32 = 1615;
pub const ERROR_BAD_RECOVERY_POLICY: i32 = 6012;
pub const ERROR_BAD_REM_ADAP: i32 = 60;
pub const ERROR_BAD_SERVICE_ENTRYPOINT: i32 = 610;
pub const ERROR_BAD_STACK: i32 = 543;
pub const ERROR_BAD_THREADID_ADDR: i32 = 159;
pub const ERROR_BAD_TOKEN_TYPE: i32 = 1349;
pub const ERROR_BAD_UNIT: i32 = 20;
pub const ERROR_BAD_USERNAME: i32 = 2202;
pub const ERROR_BAD_USER_PROFILE: i32 = 1253;
pub const ERROR_BAD_VALIDATION_CLASS: i32 = 1348;
pub const ERROR_BEGINNING_OF_MEDIA: i32 = 1102;
pub const ERROR_BEYOND_VDL: i32 = 1289;
pub const ERROR_BIOS_FAILED_TO_CONNECT_INTERRUPT: i32 = 585;
pub const ERROR_BLOCKED_BY_PARENTAL_CONTROLS: i32 = 346;
pub const ERROR_BLOCK_SHARED: i32 = 514;
pub const ERROR_BLOCK_SOURCE_WEAK_REFERENCE_INVALID: i32 = 512;
pub const ERROR_BLOCK_TARGET_WEAK_REFERENCE_INVALID: i32 = 513;
pub const ERROR_BLOCK_TOO_MANY_REFERENCES: i32 = 347;
pub const ERROR_BLOCK_WEAK_REFERENCE_INVALID: i32 = 511;
pub const ERROR_BOOT_ALREADY_ACCEPTED: i32 = 1076;
pub const ERROR_BROKEN_PIPE: i32 = 109;
pub const ERROR_BUFFER_ALL_ZEROS: i32 = 754;
pub const ERROR_BUFFER_OVERFLOW: i32 = 111;
pub const ERROR_BUSY: i32 = 170;
pub const ERROR_BUSY_DRIVE: i32 = 142;
pub const ERROR_BUS_RESET: i32 = 1111;
pub const ERROR_BYPASSIO_FLT_NOT_SUPPORTED: i32 = 506;
pub const ERROR_CACHE_PAGE_LOCKED: i32 = 752;
pub const ERROR_CALLBACK_INVOKE_INLINE: i32 = 812;
pub const ERROR_CALLBACK_POP_STACK: i32 = 768;
pub const ERROR_CALLBACK_SUPPLIED_INVALID_DATA: i32 = 1273;
pub const ERROR_CALL_NOT_IMPLEMENTED: i32 = 120;
pub const ERROR_CANCELLED: i32 = 1223;
pub const ERROR_CANCEL_VIOLATION: i32 = 173;
pub const ERROR_CANNOT_BREAK_OPLOCK: i32 = 802;
pub const ERROR_CANNOT_COPY: i32 = 266;
pub const ERROR_CANNOT_DETECT_DRIVER_FAILURE: i32 = 1080;
pub const ERROR_CANNOT_DETECT_PROCESS_ABORT: i32 = 1081;
pub const ERROR_CANNOT_FIND_WND_CLASS: i32 = 1407;
pub const ERROR_CANNOT_GRANT_REQUESTED_OPLOCK: i32 = 801;
pub const ERROR_CANNOT_IMPERSONATE: i32 = 1368;
pub const ERROR_CANNOT_LOAD_REGISTRY_FILE: i32 = 589;
pub const ERROR_CANNOT_MAKE: i32 = 82;
pub const ERROR_CANNOT_OPEN_PROFILE: i32 = 1205;
pub const ERROR_CANTFETCHBACKWARDS: i32 = 770;
pub const ERROR_CANTOPEN: i32 = 1011;
pub const ERROR_CANTREAD: i32 = 1012;
pub const ERROR_CANTSCROLLBACKWARDS: i32 = 771;
pub const ERROR_CANTWRITE: i32 = 1013;
pub const ERROR_CANT_ACCESS_DOMAIN_INFO: i32 = 1351;
pub const ERROR_CANT_ACCESS_FILE: i32 = 1920;
pub const ERROR_CANT_CLEAR_ENCRYPTION_FLAG: i32 = 432;
pub const ERROR_CANT_DISABLE_MANDATORY: i32 = 1310;
pub const ERROR_CANT_ENABLE_DENY_ONLY: i32 = 629;
pub const ERROR_CANT_OPEN_ANONYMOUS: i32 = 1347;
pub const ERROR_CANT_RESOLVE_FILENAME: i32 = 1921;
pub const ERROR_CANT_TERMINATE_SELF: i32 = 555;
pub const ERROR_CANT_WAIT: i32 = 554;
pub const ERROR_CAN_NOT_COMPLETE: i32 = 1003;
pub const ERROR_CAPAUTHZ_CHANGE_TYPE: i32 = 451;
pub const ERROR_CAPAUTHZ_DB_CORRUPTED: i32 = 455;
pub const ERROR_CAPAUTHZ_NOT_AUTHORIZED: i32 = 453;
pub const ERROR_CAPAUTHZ_NOT_DEVUNLOCKED: i32 = 450;
pub const ERROR_CAPAUTHZ_NOT_PROVISIONED: i32 = 452;
pub const ERROR_CAPAUTHZ_NO_POLICY: i32 = 454;
pub const ERROR_CAPAUTHZ_SCCD_DEV_MODE_REQUIRED: i32 = 459;
pub const ERROR_CAPAUTHZ_SCCD_INVALID_CATALOG: i32 = 456;
pub const ERROR_CAPAUTHZ_SCCD_NO_AUTH_ENTITY: i32 = 457;
pub const ERROR_CAPAUTHZ_SCCD_NO_CAPABILITY_MATCH: i32 = 460;
pub const ERROR_CAPAUTHZ_SCCD_PARSE_ERROR: i32 = 458;
pub const ERROR_CARDBUS_NOT_SUPPORTED: i32 = 724;
pub const ERROR_CASE_DIFFERING_NAMES_IN_DIR: i32 = 424;
pub const ERROR_CASE_SENSITIVE_PATH: i32 = 442;
pub const ERROR_CERTIFICATE_VALIDATION_PREFERENCE_CONFLICT: i32 = 817;
pub const ERROR_CHECKING_FILE_SYSTEM: i32 = 712;
pub const ERROR_CHECKOUT_REQUIRED: i32 = 221;
pub const ERROR_CHILD_MUST_BE_VOLATILE: i32 = 1021;
pub const ERROR_CHILD_NOT_COMPLETE: i32 = 129;
pub const ERROR_CHILD_PROCESS_BLOCKED: i32 = 367;
pub const ERROR_CHILD_WINDOW_MENU: i32 = 1436;
pub const ERROR_CIMFS_IMAGE_CORRUPT: i32 = 470;
pub const ERROR_CIMFS_IMAGE_VERSION_NOT_SUPPORTED: i32 = 471;
pub const ERROR_CIRCULAR_DEPENDENCY: i32 = 1059;
pub const ERROR_CLASS_ALREADY_EXISTS: i32 = 1410;
pub const ERROR_CLASS_DOES_NOT_EXIST: i32 = 1411;
pub const ERROR_CLASS_HAS_WINDOWS: i32 = 1412;
pub const ERROR_CLIENT_SERVER_PARAMETERS_INVALID: i32 = 597;
pub const ERROR_CLIPBOARD_NOT_OPEN: i32 = 1418;
pub const ERROR_CLOUD_FILE_ACCESS_DENIED: i32 = 395;
pub const ERROR_CLOUD_FILE_ALREADY_CONNECTED: i32 = 378;
pub const ERROR_CLOUD_FILE_AUTHENTICATION_FAILED: i32 = 386;
pub const ERROR_CLOUD_FILE_CONNECTED_PROVIDER_ONLY: i32 = 382;
pub const ERROR_CLOUD_FILE_DEHYDRATION_DISALLOWED: i32 = 434;
pub const ERROR_CLOUD_FILE_INCOMPATIBLE_HARDLINKS: i32 = 396;
pub const ERROR_CLOUD_FILE_INSUFFICIENT_RESOURCES: i32 = 387;
pub const ERROR_CLOUD_FILE_INVALID_REQUEST: i32 = 380;
pub const ERROR_CLOUD_FILE_IN_USE: i32 = 391;
pub const ERROR_CLOUD_FILE_METADATA_CORRUPT: i32 = 363;
pub const ERROR_CLOUD_FILE_METADATA_TOO_LARGE: i32 = 364;
pub const ERROR_CLOUD_FILE_NETWORK_UNAVAILABLE: i32 = 388;
pub const ERROR_CLOUD_FILE_NOT_IN_SYNC: i32 = 377;
pub const ERROR_CLOUD_FILE_NOT_SUPPORTED: i32 = 379;
pub const ERROR_CLOUD_FILE_NOT_UNDER_SYNC_ROOT: i32 = 390;
pub const ERROR_CLOUD_FILE_PINNED: i32 = 392;
pub const ERROR_CLOUD_FILE_PROPERTY_BLOB_CHECKSUM_MISMATCH: i32 = 366;
pub const ERROR_CLOUD_FILE_PROPERTY_BLOB_TOO_LARGE: i32 = 365;
pub const ERROR_CLOUD_FILE_PROPERTY_CORRUPT: i32 = 394;
pub const ERROR_CLOUD_FILE_PROPERTY_LOCK_CONFLICT: i32 = 397;
pub const ERROR_CLOUD_FILE_PROPERTY_VERSION_NOT_SUPPORTED: i32 = 375;
pub const ERROR_CLOUD_FILE_PROVIDER_NOT_RUNNING: i32 = 362;
pub const ERROR_CLOUD_FILE_PROVIDER_TERMINATED: i32 = 404;
pub const ERROR_CLOUD_FILE_READ_ONLY_VOLUME: i32 = 381;
pub const ERROR_CLOUD_FILE_REQUEST_ABORTED: i32 = 393;
pub const ERROR_CLOUD_FILE_REQUEST_CANCELED: i32 = 398;
pub const ERROR_CLOUD_FILE_REQUEST_TIMEOUT: i32 = 426;
pub const ERROR_CLOUD_FILE_SYNC_ROOT_METADATA_CORRUPT: i32 = 358;
pub const ERROR_CLOUD_FILE_TOO_MANY_PROPERTY_BLOBS: i32 = 374;
pub const ERROR_CLOUD_FILE_UNSUCCESSFUL: i32 = 389;
pub const ERROR_CLOUD_FILE_US_MESSAGE_TIMEOUT: i32 = 475;
pub const ERROR_CLOUD_FILE_VALIDATION_FAILED: i32 = 383;
pub const ERROR_COMMITMENT_LIMIT: i32 = 1455;
pub const ERROR_COMMITMENT_MINIMUM: i32 = 635;
pub const ERROR_COMPRESSED_FILE_NOT_SUPPORTED: i32 = 335;
pub const ERROR_COMPRESSION_DISABLED: i32 = 769;
pub const ERROR_COMPRESSION_NOT_BENEFICIAL: i32 = 344;
pub const ERROR_CONNECTED_OTHER_PASSWORD: i32 = 2108;
pub const ERROR_CONNECTED_OTHER_PASSWORD_DEFAULT: i32 = 2109;
pub const ERROR_CONNECTION_ABORTED: i32 = 1236;
pub const ERROR_CONNECTION_ACTIVE: i32 = 1230;
pub const ERROR_CONNECTION_COUNT_LIMIT: i32 = 1238;
pub const ERROR_CONNECTION_INVALID: i32 = 1229;
pub const ERROR_CONNECTION_REFUSED: i32 = 1225;
pub const ERROR_CONNECTION_UNAVAIL: i32 = 1201;
pub const ERROR_CONTAINER_ASSIGNED: i32 = 1504;
pub const ERROR_CONTENT_BLOCKED: i32 = 1296;
pub const ERROR_CONTEXT_EXPIRED: i32 = 1931;
pub const ERROR_CONTINUE: i32 = 1246;
pub const ERROR_CONTROL_C_EXIT: i32 = 572;
pub const ERROR_CONTROL_ID_NOT_FOUND: i32 = 1421;
pub const ERROR_CONVERT_TO_LARGE: i32 = 600;
pub const ERROR_CORRUPT_LOG_CLEARED: i32 = 798;
pub const ERROR_CORRUPT_LOG_CORRUPTED: i32 = 795;
pub const ERROR_CORRUPT_LOG_DELETED_FULL: i32 = 797;
pub const ERROR_CORRUPT_LOG_OVERFULL: i32 = 794;
pub const ERROR_CORRUPT_LOG_UNAVAILABLE: i32 = 796;
pub const ERROR_CORRUPT_SYSTEM_FILE: i32 = 634;
pub const ERROR_COULD_NOT_INTERPRET: i32 = 552;
pub const ERROR_COUNTER_TIMEOUT: i32 = 1121;
pub const ERROR_CPU_SET_INVALID: i32 = 813;
pub const ERROR_CRASH_DUMP: i32 = 753;
pub const ERROR_CRC: i32 = 23;
pub const ERROR_CREATE_FAILED: i32 = 1631;
pub const ERROR_CROSS_PARTITION_VIOLATION: i32 = 1661;
pub const ERROR_CSCSHARE_OFFLINE: i32 = 1262;
pub const ERROR_CS_ENCRYPTION_EXISTING_ENCRYPTED_FILE: i32 = 6019;
pub const ERROR_CS_ENCRYPTION_FILE_NOT_CSE: i32 = 6021;
pub const ERROR_CS_ENCRYPTION_INVALID_SERVER_RESPONSE: i32 = 6017;
pub const ERROR_CS_ENCRYPTION_NEW_ENCRYPTED_FILE: i32 = 6020;
pub const ERROR_CS_ENCRYPTION_UNSUPPORTED_SERVER: i32 = 6018;
pub const ERROR_CTX_CLIENT_QUERY_TIMEOUT: i32 = 7040;
pub const ERROR_CTX_MODEM_RESPONSE_TIMEOUT: i32 = 7012;
pub const ERROR_CURRENT_DIRECTORY: i32 = 16;
pub const ERROR_CURRENT_DOMAIN_NOT_ALLOWED: i32 = 1399;
pub const ERROR_DATABASE_DOES_NOT_EXIST: i32 = 1065;
pub const ERROR_DATATYPE_MISMATCH: i32 = 1629;
pub const ERROR_DATA_CHECKSUM_ERROR: i32 = 323;
pub const ERROR_DATA_NOT_ACCEPTED: i32 = 592;
pub const ERROR_DAX_MAPPING_EXISTS: i32 = 361;
pub const ERROR_DBG_COMMAND_EXCEPTION: i32 = 697;
pub const ERROR_DBG_CONTINUE: i32 = 767;
pub const ERROR_DBG_CONTROL_BREAK: i32 = 696;
pub const ERROR_DBG_CONTROL_C: i32 = 693;
pub const ERROR_DBG_EXCEPTION_HANDLED: i32 = 766;
pub const ERROR_DBG_EXCEPTION_NOT_HANDLED: i32 = 688;
pub const ERROR_DBG_PRINTEXCEPTION_C: i32 = 694;
pub const ERROR_DBG_REPLY_LATER: i32 = 689;
pub const ERROR_DBG_RIPEXCEPTION: i32 = 695;
pub const ERROR_DBG_TERMINATE_PROCESS: i32 = 692;
pub const ERROR_DBG_TERMINATE_THREAD: i32 = 691;
pub const ERROR_DBG_UNABLE_TO_PROVIDE_HANDLE: i32 = 690;
pub const ERROR_DC_NOT_FOUND: i32 = 1425;
pub const ERROR_DDE_FAIL: i32 = 1156;
pub const ERROR_DEBUGGER_INACTIVE: i32 = 1284;
pub const ERROR_DEBUG_ATTACH_FAILED: i32 = 590;
pub const ERROR_DECRYPTION_FAILED: i32 = 6001;
pub const ERROR_DELAY_LOAD_FAILED: i32 = 1285;
pub const ERROR_DELETE_PENDING: i32 = 303;
pub const ERROR_DEPENDENT_SERVICES_RUNNING: i32 = 1051;
pub const ERROR_DESTINATION_ELEMENT_FULL: i32 = 1161;
pub const ERROR_DESTROY_OBJECT_OF_OTHER_THREAD: i32 = 1435;
pub const ERROR_DEVICE_ALREADY_ATTACHED: i32 = 548;
pub const ERROR_DEVICE_ALREADY_REMEMBERED: i32 = 1202;
pub const ERROR_DEVICE_DOOR_OPEN: i32 = 1166;
pub const ERROR_DEVICE_ENUMERATION_ERROR: i32 = 648;
pub const ERROR_DEVICE_FEATURE_NOT_SUPPORTED: i32 = 316;
pub const ERROR_DEVICE_HARDWARE_ERROR: i32 = 483;
pub const ERROR_DEVICE_HINT_NAME_BUFFER_TOO_SMALL: i32 = 355;
pub const ERROR_DEVICE_IN_MAINTENANCE: i32 = 359;
pub const ERROR_DEVICE_IN_USE: i32 = 2404;
pub const ERROR_DEVICE_NOT_CONNECTED: i32 = 1167;
pub const ERROR_DEVICE_NOT_PARTITIONED: i32 = 1107;
pub const ERROR_DEVICE_NO_RESOURCES: i32 = 322;
pub const ERROR_DEVICE_REINITIALIZATION_NEEDED: i32 = 1164;
pub const ERROR_DEVICE_REMOVED: i32 = 1617;
pub const ERROR_DEVICE_REQUIRES_CLEANING: i32 = 1165;
pub const ERROR_DEVICE_RESET_REQUIRED: i32 = 507;
pub const ERROR_DEVICE_SUPPORT_IN_PROGRESS: i32 = 171;
pub const ERROR_DEVICE_UNREACHABLE: i32 = 321;
pub const ERROR_DEV_NOT_EXIST: i32 = 55;
pub const ERROR_DHCP_ADDRESS_CONFLICT: i32 = 4100;
pub const ERROR_DIFFERENT_SERVICE_ACCOUNT: i32 = 1079;
pub const ERROR_DIRECTORY: i32 = 267;
pub const ERROR_DIRECTORY_NOT_SUPPORTED: i32 = 336;
pub const ERROR_DIRECT_ACCESS_HANDLE: i32 = 130;
pub const ERROR_DIR_EFS_DISALLOWED: i32 = 6010;
pub const ERROR_DIR_NOT_EMPTY: i32 = 145;
pub const ERROR_DIR_NOT_ROOT: i32 = 144;
pub const ERROR_DISCARDED: i32 = 157;
pub const ERROR_DISK_CHANGE: i32 = 107;
pub const ERROR_DISK_CORRUPT: i32 = 1393;
pub const ERROR_DISK_FULL: i32 = 112;
pub const ERROR_DISK_OPERATION_FAILED: i32 = 1127;
pub const ERROR_DISK_QUOTA_EXCEEDED: i32 = 1295;
pub const ERROR_DISK_RECALIBRATE_FAILED: i32 = 1126;
pub const ERROR_DISK_REPAIR_DISABLED: i32 = 780;
pub const ERROR_DISK_REPAIR_REDIRECTED: i32 = 792;
pub const ERROR_DISK_REPAIR_UNSUCCESSFUL: i32 = 793;
pub const ERROR_DISK_RESET_FAILED: i32 = 1128;
pub const ERROR_DISK_RESOURCES_EXHAUSTED: i32 = 314;
pub const ERROR_DISK_TOO_FRAGMENTED: i32 = 302;
pub const ERROR_DLL_INIT_FAILED: i32 = 1114;
pub const ERROR_DLL_INIT_FAILED_LOGOFF: i32 = 624;
pub const ERROR_DLL_MIGHT_BE_INCOMPATIBLE: i32 = 687;
pub const ERROR_DLL_MIGHT_BE_INSECURE: i32 = 686;
pub const ERROR_DLL_NOT_FOUND: i32 = 1157;
pub const ERROR_DLP_POLICY_DENIES_OPERATION: i32 = 446;
pub const ERROR_DLP_POLICY_SILENTLY_FAIL: i32 = 449;
pub const ERROR_DLP_POLICY_WARNS_AGAINST_OPERATION: i32 = 445;
pub const ERROR_DOMAIN_CONTROLLER_EXISTS: i32 = 1250;
pub const ERROR_DOMAIN_CONTROLLER_NOT_FOUND: i32 = 1908;
pub const ERROR_DOMAIN_CTRLR_CONFIG_ERROR: i32 = 581;
pub const ERROR_DOMAIN_EXISTS: i32 = 1356;
pub const ERROR_DOMAIN_LIMIT_EXCEEDED: i32 = 1357;
pub const ERROR_DOMAIN_SID_SAME_AS_LOCAL_WORKSTATION: i32 = 8644;
pub const ERROR_DOMAIN_TRUST_INCONSISTENT: i32 = 1810;
pub const ERROR_DOWNGRADE_DETECTED: i32 = 1265;
pub const ERROR_DPL_NOT_SUPPORTED_FOR_USER: i32 = 423;
pub const ERROR_DRIVERS_LEAKING_LOCKED_PAGES: i32 = 729;
pub const ERROR_DRIVER_BLOCKED: i32 = 1275;
pub const ERROR_DRIVER_CANCEL_TIMEOUT: i32 = 594;
pub const ERROR_DRIVER_DATABASE_ERROR: i32 = 652;
pub const ERROR_DRIVER_FAILED_PRIOR_UNLOAD: i32 = 654;
pub const ERROR_DRIVER_FAILED_SLEEP: i32 = 633;
pub const ERROR_DRIVER_PROCESS_TERMINATED: i32 = 1291;
pub const ERROR_DRIVE_LOCKED: i32 = 108;
pub const ERROR_DS_ADD_REPLICA_INHIBITED: i32 = 8302;
pub const ERROR_DS_ADMIN_LIMIT_EXCEEDED: i32 = 8228;
pub const ERROR_DS_AFFECTS_MULTIPLE_DSAS: i32 = 8249;
pub const ERROR_DS_AG_CANT_HAVE_UNIVERSAL_MEMBER: i32 = 8578;
pub const ERROR_DS_ALIASED_OBJ_MISSING: i32 = 8334;
pub const ERROR_DS_ALIAS_DEREF_PROBLEM: i32 = 8244;
pub const ERROR_DS_ALIAS_POINTS_TO_ALIAS: i32 = 8336;
pub const ERROR_DS_ALIAS_PROBLEM: i32 = 8241;
pub const ERROR_DS_ATTRIBUTE_OR_VALUE_EXISTS: i32 = 8205;
pub const ERROR_DS_ATTRIBUTE_OWNED_BY_SAM: i32 = 8346;
pub const ERROR_DS_ATTRIBUTE_TYPE_UNDEFINED: i32 = 8204;
pub const ERROR_DS_ATT_ALREADY_EXISTS: i32 = 8318;
pub const ERROR_DS_ATT_IS_NOT_ON_OBJ: i32 = 8310;
pub const ERROR_DS_ATT_NOT_DEF_FOR_CLASS: i32 = 8317;
pub const ERROR_DS_ATT_NOT_DEF_IN_SCHEMA: i32 = 8303;
pub const ERROR_DS_ATT_SCHEMA_REQ_ID: i32 = 8399;
pub const ERROR_DS_ATT_SCHEMA_REQ_SYNTAX: i32 = 8416;
pub const ERROR_DS_ATT_VAL_ALREADY_EXISTS: i32 = 8323;
pub const ERROR_DS_AUDIT_FAILURE: i32 = 8625;
pub const ERROR_DS_AUTHORIZATION_FAILED: i32 = 8599;
pub const ERROR_DS_AUTH_METHOD_NOT_SUPPORTED: i32 = 8231;
pub const ERROR_DS_AUTH_UNKNOWN: i32 = 8234;
pub const ERROR_DS_AUX_CLS_TEST_FAIL: i32 = 8389;
pub const ERROR_DS_BACKLINK_WITHOUT_LINK: i32 = 8482;
pub const ERROR_DS_BAD_ATT_SCHEMA_SYNTAX: i32 = 8400;
pub const ERROR_DS_BAD_HIERARCHY_FILE: i32 = 8425;
pub const ERROR_DS_BAD_INSTANCE_TYPE: i32 = 8313;
pub const ERROR_DS_BAD_NAME_SYNTAX: i32 = 8335;
pub const ERROR_DS_BAD_RDN_ATT_ID_SYNTAX: i32 = 8392;
pub const ERROR_DS_BUILD_HIERARCHY_TABLE_FAILED: i32 = 8426;
pub const ERROR_DS_BUSY: i32 = 8206;
pub const ERROR_DS_CANT_ACCESS_REMOTE_PART_OF_AD: i32 = 8585;
pub const ERROR_DS_CANT_ADD_ATT_VALUES: i32 = 8320;
pub const ERROR_DS_CANT_ADD_SYSTEM_ONLY: i32 = 8358;
pub const ERROR_DS_CANT_ADD_TO_GC: i32 = 8550;
pub const ERROR_DS_CANT_CACHE_ATT: i32 = 8401;
pub const ERROR_DS_CANT_CACHE_CLASS: i32 = 8402;
pub const ERROR_DS_CANT_CREATE_IN_NONDOMAIN_NC: i32 = 8553;
pub const ERROR_DS_CANT_CREATE_UNDER_SCHEMA: i32 = 8510;
pub const ERROR_DS_CANT_DELETE: i32 = 8398;
pub const ERROR_DS_CANT_DELETE_DSA_OBJ: i32 = 8340;
pub const ERROR_DS_CANT_DEL_MASTER_CROSSREF: i32 = 8375;
pub const ERROR_DS_CANT_DEMOTE_WITH_WRITEABLE_NC: i32 = 8604;
pub const ERROR_DS_CANT_DEREF_ALIAS: i32 = 8337;
pub const ERROR_DS_CANT_DERIVE_SPN_FOR_DELETED_DOMAIN: i32 = 8603;
pub const ERROR_DS_CANT_DERIVE_SPN_WITHOUT_SERVER_REF: i32 = 8589;
pub const ERROR_DS_CANT_FIND_DC_FOR_SRC_DOMAIN: i32 = 8537;
pub const ERROR_DS_CANT_FIND_DSA_OBJ: i32 = 8419;
pub const ERROR_DS_CANT_FIND_EXPECTED_NC: i32 = 8420;
pub const ERROR_DS_CANT_FIND_NC_IN_CACHE: i32 = 8421;
pub const ERROR_DS_CANT_MIX_MASTER_AND_REPS: i32 = 8331;
pub const ERROR_DS_CANT_MOD_OBJ_CLASS: i32 = 8215;
pub const ERROR_DS_CANT_MOD_PRIMARYGROUPID: i32 = 8506;
pub const ERROR_DS_CANT_MOD_SYSTEM_ONLY: i32 = 8369;
pub const ERROR_DS_CANT_MOVE_ACCOUNT_GROUP: i32 = 8498;
pub const ERROR_DS_CANT_MOVE_APP_BASIC_GROUP: i32 = 8608;
pub const ERROR_DS_CANT_MOVE_APP_QUERY_GROUP: i32 = 8609;
pub const ERROR_DS_CANT_MOVE_DELETED_OBJECT: i32 = 8489;
pub const ERROR_DS_CANT_MOVE_RESOURCE_GROUP: i32 = 8499;
pub const ERROR_DS_CANT_ON_NON_LEAF: i32 = 8213;
pub const ERROR_DS_CANT_ON_RDN: i32 = 8214;
pub const ERROR_DS_CANT_REMOVE_ATT_CACHE: i32 = 8403;
pub const ERROR_DS_CANT_REMOVE_CLASS_CACHE: i32 = 8404;
pub const ERROR_DS_CANT_REM_MISSING_ATT: i32 = 8324;
pub const ERROR_DS_CANT_REM_MISSING_ATT_VAL: i32 = 8325;
pub const ERROR_DS_CANT_REPLACE_HIDDEN_REC: i32 = 8424;
pub const ERROR_DS_CANT_RETRIEVE_ATTS: i32 = 8481;
pub const ERROR_DS_CANT_RETRIEVE_CHILD: i32 = 8422;
pub const ERROR_DS_CANT_RETRIEVE_DN: i32 = 8405;
pub const ERROR_DS_CANT_RETRIEVE_INSTANCE: i32 = 8407;
pub const ERROR_DS_CANT_RETRIEVE_SD: i32 = 8526;
pub const ERROR_DS_CANT_START: i32 = 8531;
pub const ERROR_DS_CANT_TREE_DELETE_CRITICAL_OBJ: i32 = 8560;
pub const ERROR_DS_CANT_WITH_ACCT_GROUP_MEMBERSHPS: i32 = 8493;
pub const ERROR_DS_CHILDREN_EXIST: i32 = 8332;
pub const ERROR_DS_CLASS_MUST_BE_CONCRETE: i32 = 8359;
pub const ERROR_DS_CLASS_NOT_DSA: i32 = 8343;
pub const ERROR_DS_CLIENT_LOOP: i32 = 8259;
pub const ERROR_DS_CODE_INCONSISTENCY: i32 = 8408;
pub const ERROR_DS_COMPARE_FALSE: i32 = 8229;
pub const ERROR_DS_COMPARE_TRUE: i32 = 8230;
pub const ERROR_DS_CONFIDENTIALITY_REQUIRED: i32 = 8237;
pub const ERROR_DS_CONFIG_PARAM_MISSING: i32 = 8427;
pub const ERROR_DS_CONSTRAINT_VIOLATION: i32 = 8239;
pub const ERROR_DS_CONSTRUCTED_ATT_MOD: i32 = 8475;
pub const ERROR_DS_CONTROL_NOT_FOUND: i32 = 8258;
pub const ERROR_DS_COULDNT_CONTACT_FSMO: i32 = 8367;
pub const ERROR_DS_COULDNT_IDENTIFY_OBJECTS_FOR_TREE_DELETE: i32 = 8503;
pub const ERROR_DS_COULDNT_LOCK_TREE_FOR_DELETE: i32 = 8502;
pub const ERROR_DS_COULDNT_UPDATE_SPNS: i32 = 8525;
pub const ERROR_DS_COUNTING_AB_INDICES_FAILED: i32 = 8428;
pub const ERROR_DS_CROSS_DOMAIN_CLEANUP_REQD: i32 = 8491;
pub const ERROR_DS_CROSS_DOM_MOVE_ERROR: i32 = 8216;
pub const ERROR_DS_CROSS_NC_DN_RENAME: i32 = 8368;
pub const ERROR_DS_CROSS_REF_BUSY: i32 = 8602;
pub const ERROR_DS_CROSS_REF_EXISTS: i32 = 8374;
pub const ERROR_DS_CR_IMPOSSIBLE_TO_VALIDATE: i32 = 8495;
pub const ERROR_DS_CR_IMPOSSIBLE_TO_VALIDATE_V2: i32 = 8586;
pub const ERROR_DS_DATABASE_ERROR: i32 = 8409;
pub const ERROR_DS_DECODING_ERROR: i32 = 8253;
pub const ERROR_DS_DESTINATION_AUDITING_NOT_ENABLED: i32 = 8536;
pub const ERROR_DS_DESTINATION_DOMAIN_NOT_IN_FOREST: i32 = 8535;
pub const ERROR_DS_DIFFERENT_REPL_EPOCHS: i32 = 8593;
pub const ERROR_DS_DISALLOWED_IN_SYSTEM_CONTAINER: i32 = 8615;
pub const ERROR_DS_DISALLOWED_NC_REDIRECT: i32 = 8640;
pub const ERROR_DS_DNS_LOOKUP_FAILURE: i32 = 8524;
pub const ERROR_DS_DOMAIN_NAME_EXISTS_IN_FOREST: i32 = 8634;
pub const ERROR_DS_DOMAIN_RENAME_IN_PROGRESS: i32 = 8612;
pub const ERROR_DS_DOMAIN_VERSION_TOO_HIGH: i32 = 8564;
pub const ERROR_DS_DOMAIN_VERSION_TOO_LOW: i32 = 8566;
pub const ERROR_DS_DRA_ABANDON_SYNC: i32 = 8462;
pub const ERROR_DS_DRA_ACCESS_DENIED: i32 = 8453;
pub const ERROR_DS_DRA_BAD_DN: i32 = 8439;
pub const ERROR_DS_DRA_BAD_INSTANCE_TYPE: i32 = 8445;
pub const ERROR_DS_DRA_BAD_NC: i32 = 8440;
pub const ERROR_DS_DRA_BUSY: i32 = 8438;
pub const ERROR_DS_DRA_CONNECTION_FAILED: i32 = 8444;
pub const ERROR_DS_DRA_CORRUPT_UTD_VECTOR: i32 = 8629;
pub const ERROR_DS_DRA_DB_ERROR: i32 = 8451;
pub const ERROR_DS_DRA_DN_EXISTS: i32 = 8441;
pub const ERROR_DS_DRA_EARLIER_SCHEMA_CONFLICT: i32 = 8544;
pub const ERROR_DS_DRA_EXTN_CONNECTION_FAILED: i32 = 8466;
pub const ERROR_DS_DRA_GENERIC: i32 = 8436;
pub const ERROR_DS_DRA_INCOMPATIBLE_PARTIAL_SET: i32 = 8464;
pub const ERROR_DS_DRA_INCONSISTENT_DIT: i32 = 8443;
pub const ERROR_DS_DRA_INTERNAL_ERROR: i32 = 8442;
pub const ERROR_DS_DRA_INVALID_PARAMETER: i32 = 8437;
pub const ERROR_DS_DRA_MAIL_PROBLEM: i32 = 8447;
pub const ERROR_DS_DRA_MISSING_KRBTGT_SECRET: i32 = 8633;
pub const ERROR_DS_DRA_MISSING_PARENT: i32 = 8460;
pub const ERROR_DS_DRA_NAME_COLLISION: i32 = 8458;
pub const ERROR_DS_DRA_NOT_SUPPORTED: i32 = 8454;
pub const ERROR_DS_DRA_NO_REPLICA: i32 = 8452;
pub const ERROR_DS_DRA_OBJ_IS_REP_SOURCE: i32 = 8450;
pub const ERROR_DS_DRA_OBJ_NC_MISMATCH: i32 = 8545;
pub const ERROR_DS_DRA_OUT_OF_MEM: i32 = 8446;
pub const ERROR_DS_DRA_OUT_SCHEDULE_WINDOW: i32 = 8617;
pub const ERROR_DS_DRA_PREEMPTED: i32 = 8461;
pub const ERROR_DS_DRA_RECYCLED_TARGET: i32 = 8639;
pub const ERROR_DS_DRA_REF_ALREADY_EXISTS: i32 = 8448;
pub const ERROR_DS_DRA_REF_NOT_FOUND: i32 = 8449;
pub const ERROR_DS_DRA_REPL_PENDING: i32 = 8477;
pub const ERROR_DS_DRA_RPC_CANCELLED: i32 = 8455;
pub const ERROR_DS_DRA_SCHEMA_CONFLICT: i32 = 8543;
pub const ERROR_DS_DRA_SCHEMA_INFO_SHIP: i32 = 8542;
pub const ERROR_DS_DRA_SCHEMA_MISMATCH: i32 = 8418;
pub const ERROR_DS_DRA_SECRETS_DENIED: i32 = 8630;
pub const ERROR_DS_DRA_SHUTDOWN: i32 = 8463;
pub const ERROR_DS_DRA_SINK_DISABLED: i32 = 8457;
pub const ERROR_DS_DRA_SOURCE_DISABLED: i32 = 8456;
pub const ERROR_DS_DRA_SOURCE_IS_PARTIAL_REPLICA: i32 = 8465;
pub const ERROR_DS_DRA_SOURCE_REINSTALLED: i32 = 8459;
pub const ERROR_DS_DRS_EXTENSIONS_CHANGED: i32 = 8594;
pub const ERROR_DS_DSA_MUST_BE_INT_MASTER: i32 = 8342;
pub const ERROR_DS_DST_DOMAIN_NOT_NATIVE: i32 = 8496;
pub const ERROR_DS_DST_NC_MISMATCH: i32 = 8486;
pub const ERROR_DS_DS_REQUIRED: i32 = 8478;
pub const ERROR_DS_DUPLICATE_ID_FOUND: i32 = 8605;
pub const ERROR_DS_DUP_LDAP_DISPLAY_NAME: i32 = 8382;
pub const ERROR_DS_DUP_LINK_ID: i32 = 8468;
pub const ERROR_DS_DUP_MAPI_ID: i32 = 8380;
pub const ERROR_DS_DUP_MSDS_INTID: i32 = 8597;
pub const ERROR_DS_DUP_OID: i32 = 8379;
pub const ERROR_DS_DUP_RDN: i32 = 8378;
pub const ERROR_DS_DUP_SCHEMA_ID_GUID: i32 = 8381;
pub const ERROR_DS_ENCODING_ERROR: i32 = 8252;
pub const ERROR_DS_EPOCH_MISMATCH: i32 = 8483;
pub const ERROR_DS_EXISTING_AD_CHILD_NC: i32 = 8613;
pub const ERROR_DS_EXISTS_IN_AUX_CLS: i32 = 8393;
pub const ERROR_DS_EXISTS_IN_MAY_HAVE: i32 = 8386;
pub const ERROR_DS_EXISTS_IN_MUST_HAVE: i32 = 8385;
pub const ERROR_DS_EXISTS_IN_POSS_SUP: i32 = 8395;
pub const ERROR_DS_EXISTS_IN_RDNATTID: i32 = 8598;
pub const ERROR_DS_EXISTS_IN_SUB_CLS: i32 = 8394;
pub const ERROR_DS_FILTER_UNKNOWN: i32 = 8254;
pub const ERROR_DS_FILTER_USES_CONTRUCTED_ATTRS: i32 = 8555;
pub const ERROR_DS_FLAT_NAME_EXISTS_IN_FOREST: i32 = 8635;
pub const ERROR_DS_FOREST_VERSION_TOO_HIGH: i32 = 8563;
pub const ERROR_DS_FOREST_VERSION_TOO_LOW: i32 = 8565;
pub const ERROR_DS_GCVERIFY_ERROR: i32 = 8417;
pub const ERROR_DS_GC_NOT_AVAILABLE: i32 = 8217;
pub const ERROR_DS_GC_REQUIRED: i32 = 8547;
pub const ERROR_DS_GENERIC_ERROR: i32 = 8341;
pub const ERROR_DS_GLOBAL_CANT_HAVE_CROSSDOMAIN_MEMBER: i32 = 8519;
pub const ERROR_DS_GLOBAL_CANT_HAVE_LOCAL_MEMBER: i32 = 8516;
pub const ERROR_DS_GLOBAL_CANT_HAVE_UNIVERSAL_MEMBER: i32 = 8517;
pub const ERROR_DS_GOVERNSID_MISSING: i32 = 8410;
pub const ERROR_DS_GROUP_CONVERSION_ERROR: i32 = 8607;
pub const ERROR_DS_HAVE_PRIMARY_MEMBERS: i32 = 8521;
pub const ERROR_DS_HIERARCHY_TABLE_MALLOC_FAILED: i32 = 8429;
pub const ERROR_DS_HIERARCHY_TABLE_TOO_DEEP: i32 = 8628;
pub const ERROR_DS_HIGH_ADLDS_FFL: i32 = 8641;
pub const ERROR_DS_HIGH_DSA_VERSION: i32 = 8642;
pub const ERROR_DS_ILLEGAL_BASE_SCHEMA_MOD: i32 = 8507;
pub const ERROR_DS_ILLEGAL_MOD_OPERATION: i32 = 8311;
pub const ERROR_DS_ILLEGAL_SUPERIOR: i32 = 8345;
pub const ERROR_DS_ILLEGAL_XDOM_MOVE_OPERATION: i32 = 8492;
pub const ERROR_DS_INAPPROPRIATE_AUTH: i32 = 8233;
pub const ERROR_DS_INAPPROPRIATE_MATCHING: i32 = 8238;
pub const ERROR_DS_INCOMPATIBLE_CONTROLS_USED: i32 = 8574;
pub const ERROR_DS_INCOMPATIBLE_VERSION: i32 = 8567;
pub const ERROR_DS_INCORRECT_ROLE_OWNER: i32 = 8210;
pub const ERROR_DS_INIT_FAILURE: i32 = 8532;
pub const ERROR_DS_INIT_FAILURE_CONSOLE: i32 = 8561;
pub const ERROR_DS_INSTALL_NO_SCH_VERSION_IN_INIFILE: i32 = 8512;
pub const ERROR_DS_INSTALL_NO_SRC_SCH_VERSION: i32 = 8511;
pub const ERROR_DS_INSTALL_SCHEMA_MISMATCH: i32 = 8467;
pub const ERROR_DS_INSUFFICIENT_ATTR_TO_CREATE_OBJECT: i32 = 8606;
pub const ERROR_DS_INSUFF_ACCESS_RIGHTS: i32 = 8344;
pub const ERROR_DS_INTERNAL_FAILURE: i32 = 8430;
pub const ERROR_DS_INVALID_ATTRIBUTE_SYNTAX: i32 = 8203;
pub const ERROR_DS_INVALID_DMD: i32 = 8360;
pub const ERROR_DS_INVALID_DN_SYNTAX: i32 = 8242;
pub const ERROR_DS_INVALID_GROUP_TYPE: i32 = 8513;
pub const ERROR_DS_INVALID_LDAP_DISPLAY_NAME: i32 = 8479;
pub const ERROR_DS_INVALID_NAME_FOR_SPN: i32 = 8554;
pub const ERROR_DS_INVALID_ROLE_OWNER: i32 = 8366;
pub const ERROR_DS_INVALID_SCRIPT: i32 = 8600;
pub const ERROR_DS_INVALID_SEARCH_FLAG: i32 = 8500;
pub const ERROR_DS_INVALID_SEARCH_FLAG_SUBTREE: i32 = 8626;
pub const ERROR_DS_INVALID_SEARCH_FLAG_TUPLE: i32 = 8627;
pub const ERROR_DS_IS_LEAF: i32 = 8243;
pub const ERROR_DS_KEY_NOT_UNIQUE: i32 = 8527;
pub const ERROR_DS_LDAP_SEND_QUEUE_FULL: i32 = 8616;
pub const ERROR_DS_LINK_ID_NOT_AVAILABLE: i32 = 8577;
pub const ERROR_DS_LOCAL_CANT_HAVE_CROSSDOMAIN_LOCAL_MEMBER: i32 = 8520;
pub const ERROR_DS_LOCAL_ERROR: i32 = 8251;
pub const ERROR_DS_LOCAL_MEMBER_OF_LOCAL_ONLY: i32 = 8548;
pub const ERROR_DS_LOOP_DETECT: i32 = 8246;
pub const ERROR_DS_LOW_ADLDS_FFL: i32 = 8643;
pub const ERROR_DS_LOW_DSA_VERSION: i32 = 8568;
pub const ERROR_DS_MACHINE_ACCOUNT_CREATED_PRENT4: i32 = 8572;
pub const ERROR_DS_MACHINE_ACCOUNT_QUOTA_EXCEEDED: i32 = 8557;
pub const ERROR_DS_MAPI_ID_NOT_AVAILABLE: i32 = 8632;
pub const ERROR_DS_MASTERDSA_REQUIRED: i32 = 8314;
pub const ERROR_DS_MAX_OBJ_SIZE_EXCEEDED: i32 = 8304;
pub const ERROR_DS_MEMBERSHIP_EVALUATED_LOCALLY: i32 = 8201;
pub const ERROR_DS_MISSING_EXPECTED_ATT: i32 = 8411;
pub const ERROR_DS_MISSING_FOREST_TRUST: i32 = 8649;
pub const ERROR_DS_MISSING_FSMO_SETTINGS: i32 = 8434;
pub const ERROR_DS_MISSING_INFRASTRUCTURE_CONTAINER: i32 = 8497;
pub const ERROR_DS_MISSING_REQUIRED_ATT: i32 = 8316;
pub const ERROR_DS_MISSING_SUPREF: i32 = 8406;
pub const ERROR_DS_MODIFYDN_DISALLOWED_BY_FLAG: i32 = 8581;
pub const ERROR_DS_MODIFYDN_DISALLOWED_BY_INSTANCE_TYPE: i32 = 8579;
pub const ERROR_DS_MODIFYDN_WRONG_GRANDPARENT: i32 = 8582;
pub const ERROR_DS_MUST_BE_RUN_ON_DST_DC: i32 = 8558;
pub const ERROR_DS_NAME_ERROR_DOMAIN_ONLY: i32 = 8473;
pub const ERROR_DS_NAME_ERROR_NOT_FOUND: i32 = 8470;
pub const ERROR_DS_NAME_ERROR_NOT_UNIQUE: i32 = 8471;
pub const ERROR_DS_NAME_ERROR_NO_MAPPING: i32 = 8472;
pub const ERROR_DS_NAME_ERROR_NO_SYNTACTICAL_MAPPING: i32 = 8474;
pub const ERROR_DS_NAME_ERROR_RESOLVING: i32 = 8469;
pub const ERROR_DS_NAME_ERROR_TRUST_REFERRAL: i32 = 8583;
pub const ERROR_DS_NAME_NOT_UNIQUE: i32 = 8571;
pub const ERROR_DS_NAME_REFERENCE_INVALID: i32 = 8373;
pub const ERROR_DS_NAME_TOO_LONG: i32 = 8348;
pub const ERROR_DS_NAME_TOO_MANY_PARTS: i32 = 8347;
pub const ERROR_DS_NAME_TYPE_UNKNOWN: i32 = 8351;
pub const ERROR_DS_NAME_UNPARSEABLE: i32 = 8350;
pub const ERROR_DS_NAME_VALUE_TOO_LONG: i32 = 8349;
pub const ERROR_DS_NAMING_MASTER_GC: i32 = 8523;
pub const ERROR_DS_NAMING_VIOLATION: i32 = 8247;
pub const ERROR_DS_NCNAME_MISSING_CR_REF: i32 = 8412;
pub const ERROR_DS_NCNAME_MUST_BE_NC: i32 = 8357;
pub const ERROR_DS_NC_MUST_HAVE_NC_PARENT: i32 = 8494;
pub const ERROR_DS_NC_STILL_HAS_DSAS: i32 = 8546;
pub const ERROR_DS_NONEXISTENT_MAY_HAVE: i32 = 8387;
pub const ERROR_DS_NONEXISTENT_MUST_HAVE: i32 = 8388;
pub const ERROR_DS_NONEXISTENT_POSS_SUP: i32 = 8390;
pub const ERROR_DS_NONSAFE_SCHEMA_CHANGE: i32 = 8508;
pub const ERROR_DS_NON_ASQ_SEARCH: i32 = 8624;
pub const ERROR_DS_NON_BASE_SEARCH: i32 = 8480;
pub const ERROR_DS_NOTIFY_FILTER_TOO_COMPLEX: i32 = 8377;
pub const ERROR_DS_NOT_AN_OBJECT: i32 = 8352;
pub const ERROR_DS_NOT_AUTHORITIVE_FOR_DST_NC: i32 = 8487;
pub const ERROR_DS_NOT_CLOSEST: i32 = 8588;
pub const ERROR_DS_NOT_INSTALLED: i32 = 8200;
pub const ERROR_DS_NOT_ON_BACKLINK: i32 = 8362;
pub const ERROR_DS_NOT_SUPPORTED: i32 = 8256;
pub const ERROR_DS_NOT_SUPPORTED_SORT_ORDER: i32 = 8570;
pub const ERROR_DS_NO_ATTRIBUTE_OR_VALUE: i32 = 8202;
pub const ERROR_DS_NO_BEHAVIOR_VERSION_IN_MIXEDDOMAIN: i32 = 8569;
pub const ERROR_DS_NO_CHAINED_EVAL: i32 = 8328;
pub const ERROR_DS_NO_CHAINING: i32 = 8327;
pub const ERROR_DS_NO_CHECKPOINT_WITH_PDC: i32 = 8551;
pub const ERROR_DS_NO_CROSSREF_FOR_NC: i32 = 8363;
pub const ERROR_DS_NO_DELETED_NAME: i32 = 8355;
pub const ERROR_DS_NO_FPO_IN_UNIVERSAL_GROUPS: i32 = 8549;
pub const ERROR_DS_NO_MORE_RIDS: i32 = 8209;
pub const ERROR_DS_NO_MSDS_INTID: i32 = 8596;
pub const ERROR_DS_NO_NEST_GLOBALGROUP_IN_MIXEDDOMAIN: i32 = 8514;
pub const ERROR_DS_NO_NEST_LOCALGROUP_IN_MIXEDDOMAIN: i32 = 8515;
pub const ERROR_DS_NO_NTDSA_OBJECT: i32 = 8623;
pub const ERROR_DS_NO_OBJECT_MOVE_IN_SCHEMA_NC: i32 = 8580;
pub const ERROR_DS_NO_PARENT_OBJECT: i32 = 8329;
pub const ERROR_DS_NO_PKT_PRIVACY_ON_CONNECTION: i32 = 8533;
pub const ERROR_DS_NO_RDN_DEFINED_IN_SCHEMA: i32 = 8306;
pub const ERROR_DS_NO_REF_DOMAIN: i32 = 8575;
pub const ERROR_DS_NO_REQUESTED_ATTS_FOUND: i32 = 8308;
pub const ERROR_DS_NO_RESULTS_RETURNED: i32 = 8257;
pub const ERROR_DS_NO_RIDS_ALLOCATED: i32 = 8208;
pub const ERROR_DS_NO_SERVER_OBJECT: i32 = 8622;
pub const ERROR_DS_NO_SUCH_OBJECT: i32 = 8240;
pub const ERROR_DS_NO_TREE_DELETE_ABOVE_NC: i32 = 8501;
pub const ERROR_DS_NTDSCRIPT_PROCESS_ERROR: i32 = 8592;
pub const ERROR_DS_NTDSCRIPT_SYNTAX_ERROR: i32 = 8591;
pub const ERROR_DS_OBJECT_BEING_REMOVED: i32 = 8339;
pub const ERROR_DS_OBJECT_CLASS_REQUIRED: i32 = 8315;
pub const ERROR_DS_OBJECT_RESULTS_TOO_LARGE: i32 = 8248;
pub const ERROR_DS_OBJ_CLASS_NOT_DEFINED: i32 = 8371;
pub const ERROR_DS_OBJ_CLASS_NOT_SUBCLASS: i32 = 8372;
pub const ERROR_DS_OBJ_CLASS_VIOLATION: i32 = 8212;
pub const ERROR_DS_OBJ_GUID_EXISTS: i32 = 8361;
pub const ERROR_DS_OBJ_NOT_FOUND: i32 = 8333;
pub const ERROR_DS_OBJ_STRING_NAME_EXISTS: i32 = 8305;
pub const ERROR_DS_OBJ_TOO_LARGE: i32 = 8312;
pub const ERROR_DS_OFFSET_RANGE_ERROR: i32 = 8262;
pub const ERROR_DS_OID_MAPPED_GROUP_CANT_HAVE_MEMBERS: i32 = 8637;
pub const ERROR_DS_OID_NOT_FOUND: i32 = 8638;
pub const ERROR_DS_OPERATIONS_ERROR: i32 = 8224;
pub const ERROR_DS_OUT_OF_SCOPE: i32 = 8338;
pub const ERROR_DS_OUT_OF_VERSION_STORE: i32 = 8573;
pub const ERROR_DS_PARAM_ERROR: i32 = 8255;
pub const ERROR_DS_PARENT_IS_AN_ALIAS: i32 = 8330;
pub const ERROR_DS_PDC_OPERATION_IN_PROGRESS: i32 = 8490;
pub const ERROR_DS_PER_ATTRIBUTE_AUTHZ_FAILED_DURING_ADD: i32 = 8652;
pub const ERROR_DS_POLICY_NOT_KNOWN: i32 = 8618;
pub const ERROR_DS_PROTOCOL_ERROR: i32 = 8225;
pub const ERROR_DS_RANGE_CONSTRAINT: i32 = 8322;
pub const ERROR_DS_RDN_DOESNT_MATCH_SCHEMA: i32 = 8307;
pub const ERROR_DS_RECALCSCHEMA_FAILED: i32 = 8396;
pub const ERROR_DS_REFERRAL: i32 = 8235;
pub const ERROR_DS_REFERRAL_LIMIT_EXCEEDED: i32 = 8260;
pub const ERROR_DS_REFUSING_FSMO_ROLES: i32 = 8433;
pub const ERROR_DS_REMOTE_CROSSREF_OP_FAILED: i32 = 8601;
pub const ERROR_DS_REPLICATOR_ONLY: i32 = 8370;
pub const ERROR_DS_REPLICA_SET_CHANGE_NOT_ALLOWED_ON_DISABLED_CR: i32 = 8595;
pub const ERROR_DS_REPL_LIFETIME_EXCEEDED: i32 = 8614;
pub const ERROR_DS_RESERVED_LINK_ID: i32 = 8576;
pub const ERROR_DS_RESERVED_MAPI_ID: i32 = 8631;
pub const ERROR_DS_RIDMGR_DISABLED: i32 = 8263;
pub const ERROR_DS_RIDMGR_INIT_ERROR: i32 = 8211;
pub const ERROR_DS_ROLE_NOT_VERIFIED: i32 = 8610;
pub const ERROR_DS_ROOT_CANT_BE_SUBREF: i32 = 8326;
pub const ERROR_DS_ROOT_MUST_BE_NC: i32 = 8301;
pub const ERROR_DS_ROOT_REQUIRES_CLASS_TOP: i32 = 8432;
pub const ERROR_DS_SAM_INIT_FAILURE: i32 = 8504;
pub const ERROR_DS_SAM_INIT_FAILURE_CONSOLE: i32 = 8562;
pub const ERROR_DS_SAM_NEED_BOOTKEY_FLOPPY: i32 = 8530;
pub const ERROR_DS_SAM_NEED_BOOTKEY_PASSWORD: i32 = 8529;
pub const ERROR_DS_SCHEMA_ALLOC_FAILED: i32 = 8415;
pub const ERROR_DS_SCHEMA_NOT_LOADED: i32 = 8414;
pub const ERROR_DS_SCHEMA_UPDATE_DISALLOWED: i32 = 8509;
pub const ERROR_DS_SECURITY_CHECKING_ERROR: i32 = 8413;
pub const ERROR_DS_SECURITY_ILLEGAL_MODIFY: i32 = 8423;
pub const ERROR_DS_SEC_DESC_INVALID: i32 = 8354;
pub const ERROR_DS_SEC_DESC_TOO_SHORT: i32 = 8353;
pub const ERROR_DS_SEMANTIC_ATT_TEST: i32 = 8383;
pub const ERROR_DS_SENSITIVE_GROUP_VIOLATION: i32 = 8505;
pub const ERROR_DS_SERVER_DOWN: i32 = 8250;
pub const ERROR_DS_SHUTTING_DOWN: i32 = 8364;
pub const ERROR_DS_SINGLE_USER_MODE_FAILED: i32 = 8590;
pub const ERROR_DS_SINGLE_VALUE_CONSTRAINT: i32 = 8321;
pub const ERROR_DS_SIZELIMIT_EXCEEDED: i32 = 8227;
pub const ERROR_DS_SORT_CONTROL_MISSING: i32 = 8261;
pub const ERROR_DS_SOURCE_AUDITING_NOT_ENABLED: i32 = 8552;
pub const ERROR_DS_SOURCE_DOMAIN_IN_FOREST: i32 = 8534;
pub const ERROR_DS_SPN_VALUE_NOT_UNIQUE_IN_FOREST: i32 = 8647;
pub const ERROR_DS_SRC_AND_DST_NC_IDENTICAL: i32 = 8485;
pub const ERROR_DS_SRC_AND_DST_OBJECT_CLASS_MISMATCH: i32 = 8540;
pub const ERROR_DS_SRC_DC_MUST_BE_SP4_OR_GREATER: i32 = 8559;
pub const ERROR_DS_SRC_GUID_MISMATCH: i32 = 8488;
pub const ERROR_DS_SRC_NAME_MISMATCH: i32 = 8484;
pub const ERROR_DS_SRC_OBJ_NOT_GROUP_OR_USER: i32 = 8538;
pub const ERROR_DS_SRC_SID_EXISTS_IN_FOREST: i32 = 8539;
pub const ERROR_DS_STRING_SD_CONVERSION_FAILED: i32 = 8522;
pub const ERROR_DS_STRONG_AUTH_REQUIRED: i32 = 8232;
pub const ERROR_DS_SUBREF_MUST_HAVE_PARENT: i32 = 8356;
pub const ERROR_DS_SUBTREE_NOTIFY_NOT_NC_HEAD: i32 = 8376;
pub const ERROR_DS_SUB_CLS_TEST_FAIL: i32 = 8391;
pub const ERROR_DS_SYNTAX_MISMATCH: i32 = 8384;
pub const ERROR_DS_THREAD_LIMIT_EXCEEDED: i32 = 8587;
pub const ERROR_DS_TIMELIMIT_EXCEEDED: i32 = 8226;
pub const ERROR_DS_TREE_DELETE_NOT_FINISHED: i32 = 8397;
pub const ERROR_DS_UNABLE_TO_SURRENDER_ROLES: i32 = 8435;
pub const ERROR_DS_UNAVAILABLE: i32 = 8207;
pub const ERROR_DS_UNAVAILABLE_CRIT_EXTENSION: i32 = 8236;
pub const ERROR_DS_UNDELETE_SAM_VALIDATION_FAILED: i32 = 8645;
pub const ERROR_DS_UNICODEPWD_NOT_IN_QUOTES: i32 = 8556;
pub const ERROR_DS_UNIVERSAL_CANT_HAVE_LOCAL_MEMBER: i32 = 8518;
pub const ERROR_DS_UNKNOWN_ERROR: i32 = 8431;
pub const ERROR_DS_UNKNOWN_OPERATION: i32 = 8365;
pub const ERROR_DS_UNWILLING_TO_PERFORM: i32 = 8245;
pub const ERROR_DS_UPN_VALUE_NOT_UNIQUE_IN_FOREST: i32 = 8648;
pub const ERROR_DS_USER_BUFFER_TO_SMALL: i32 = 8309;
pub const ERROR_DS_VALUE_KEY_NOT_UNIQUE: i32 = 8650;
pub const ERROR_DS_VERSION_CHECK_FAILURE: i32 = 643;
pub const ERROR_DS_WKO_CONTAINER_CANNOT_BE_SPECIAL: i32 = 8611;
pub const ERROR_DS_WRONG_LINKED_ATT_SYNTAX: i32 = 8528;
pub const ERROR_DS_WRONG_OM_OBJ_CLASS: i32 = 8476;
pub const ERROR_DUPLICATE_PRIVILEGES: i32 = 311;
pub const ERROR_DUPLICATE_SERVICE_NAME: i32 = 1078;
pub const ERROR_DUP_DOMAINNAME: i32 = 1221;
pub const ERROR_DUP_NAME: i32 = 52;
pub const ERROR_DYNAMIC_CODE_BLOCKED: i32 = 1655;
pub const ERROR_DYNLINK_FROM_INVALID_RING: i32 = 196;
pub const ERROR_EAS_DIDNT_FIT: i32 = 275;
pub const ERROR_EAS_NOT_SUPPORTED: i32 = 282;
pub const ERROR_EA_ACCESS_DENIED: i32 = 994;
pub const ERROR_EA_FILE_CORRUPT: i32 = 276;
pub const ERROR_EA_LIST_INCONSISTENT: i32 = 255;
pub const ERROR_EA_TABLE_FULL: i32 = 277;
pub const ERROR_EDP_DPL_POLICY_CANT_BE_SATISFIED: i32 = 357;
pub const ERROR_EDP_POLICY_DENIES_OPERATION: i32 = 356;
pub const ERROR_EFS_ALG_BLOB_TOO_BIG: i32 = 6013;
pub const ERROR_EFS_DISABLED: i32 = 6015;
pub const ERROR_EFS_SERVER_NOT_TRUSTED: i32 = 6011;
pub const ERROR_EFS_VERSION_NOT_SUPPORT: i32 = 6016;
pub const ERROR_ELEVATION_REQUIRED: i32 = 740;
pub const ERROR_ENCLAVE_FAILURE: i32 = 349;
pub const ERROR_ENCLAVE_NOT_TERMINATED: i32 = 814;
pub const ERROR_ENCLAVE_VIOLATION: i32 = 815;
pub const ERROR_ENCRYPTED_FILE_NOT_SUPPORTED: i32 = 489;
pub const ERROR_ENCRYPTED_IO_NOT_POSSIBLE: i32 = 808;
pub const ERROR_ENCRYPTING_METADATA_DISALLOWED: i32 = 431;
pub const ERROR_ENCRYPTION_DISABLED: i32 = 430;
pub const ERROR_ENCRYPTION_FAILED: i32 = 6000;
pub const ERROR_ENCRYPTION_POLICY_DENIES_OPERATION: i32 = 6022;
pub const ERROR_END_OF_MEDIA: i32 = 1100;
pub const ERROR_ENVVAR_NOT_FOUND: i32 = 203;
pub const ERROR_EOM_OVERFLOW: i32 = 1129;
pub const ERROR_ERRORS_ENCOUNTERED: i32 = 774;
pub const ERROR_EVALUATION_EXPIRATION: i32 = 622;
pub const ERROR_EVENTLOG_CANT_START: i32 = 1501;
pub const ERROR_EVENTLOG_FILE_CHANGED: i32 = 1503;
pub const ERROR_EVENTLOG_FILE_CORRUPT: i32 = 1500;
pub const ERROR_EVENT_DONE: i32 = 710;
pub const ERROR_EVENT_PENDING: i32 = 711;
pub const ERROR_EXCEPTION_IN_SERVICE: i32 = 1064;
pub const ERROR_EXCL_SEM_ALREADY_OWNED: i32 = 101;
pub const ERROR_EXE_CANNOT_MODIFY_SIGNED_BINARY: i32 = 217;
pub const ERROR_EXE_CANNOT_MODIFY_STRONG_SIGNED_BINARY: i32 = 218;
pub const ERROR_EXE_MACHINE_TYPE_MISMATCH: i32 = 216;
pub const ERROR_EXE_MARKED_INVALID: i32 = 192;
pub const ERROR_EXTENDED_ERROR: i32 = 1208;
pub const ERROR_EXTERNAL_BACKING_PROVIDER_UNKNOWN: i32 = 343;
pub const ERROR_EXTERNAL_SYSKEY_NOT_SUPPORTED: i32 = 399;
pub const ERROR_EXTRANEOUS_INFORMATION: i32 = 677;
pub const ERROR_FAILED_DRIVER_ENTRY: i32 = 647;
pub const ERROR_FAILED_SERVICE_CONTROLLER_CONNECT: i32 = 1063;
pub const ERROR_FAIL_FAST_EXCEPTION: i32 = 1653;
pub const ERROR_FAIL_I24: i32 = 83;
pub const ERROR_FAIL_NOACTION_REBOOT: i32 = 350;
pub const ERROR_FAIL_RESTART: i32 = 352;
pub const ERROR_FAIL_SHUTDOWN: i32 = 351;
pub const ERROR_FATAL_APP_EXIT: i32 = 713;
pub const ERROR_FILEMARK_DETECTED: i32 = 1101;
pub const ERROR_FILENAME_EXCED_RANGE: i32 = 206;
pub const ERROR_FILE_CHECKED_OUT: i32 = 220;
pub const ERROR_FILE_CORRUPT: i32 = 1392;
pub const ERROR_FILE_ENCRYPTED: i32 = 6002;
pub const ERROR_FILE_EXISTS: i32 = 80;
pub const ERROR_FILE_HANDLE_REVOKED: i32 = 806;
pub const ERROR_FILE_INVALID: i32 = 1006;
pub const ERROR_FILE_LEVEL_TRIM_NOT_SUPPORTED: i32 = 326;
pub const ERROR_FILE_METADATA_OPTIMIZATION_IN_PROGRESS: i32 = 809;
pub const ERROR_FILE_NOT_ENCRYPTED: i32 = 6007;
pub const ERROR_FILE_NOT_FOUND: i32 = 2;
pub const ERROR_FILE_NOT_SUPPORTED: i32 = 425;
pub const ERROR_FILE_OFFLINE: i32 = 4350;
pub const ERROR_FILE_PROTECTED_UNDER_DPL: i32 = 406;
pub const ERROR_FILE_READ_ONLY: i32 = 6009;
pub const ERROR_FILE_SNAP_INVALID_PARAMETER: i32 = 440;
pub const ERROR_FILE_SNAP_IN_PROGRESS: i32 = 435;
pub const ERROR_FILE_SNAP_IO_NOT_COORDINATED: i32 = 438;
pub const ERROR_FILE_SNAP_MODIFY_NOT_SUPPORTED: i32 = 437;
pub const ERROR_FILE_SNAP_UNEXPECTED_ERROR: i32 = 439;
pub const ERROR_FILE_SNAP_USER_SECTION_NOT_SUPPORTED: i32 = 436;
pub const ERROR_FILE_SYSTEM_LIMITATION: i32 = 665;
pub const ERROR_FILE_SYSTEM_VIRTUALIZATION_BUSY: i32 = 371;
pub const ERROR_FILE_SYSTEM_VIRTUALIZATION_INVALID_OPERATION: i32 = 385;
pub const ERROR_FILE_SYSTEM_VIRTUALIZATION_METADATA_CORRUPT: i32 = 370;
pub const ERROR_FILE_SYSTEM_VIRTUALIZATION_PROVIDER_UNKNOWN: i32 = 372;
pub const ERROR_FILE_SYSTEM_VIRTUALIZATION_UNAVAILABLE: i32 = 369;
pub const ERROR_FILE_TOO_LARGE: i32 = 223;
pub const ERROR_FIRMWARE_UPDATED: i32 = 728;
pub const ERROR_FLOAT_MULTIPLE_FAULTS: i32 = 630;
pub const ERROR_FLOAT_MULTIPLE_TRAPS: i32 = 631;
pub const ERROR_FLOPPY_BAD_REGISTERS: i32 = 1125;
pub const ERROR_FLOPPY_ID_MARK_NOT_FOUND: i32 = 1122;
pub const ERROR_FLOPPY_UNKNOWN_ERROR: i32 = 1124;
pub const ERROR_FLOPPY_VOLUME: i32 = 584;
pub const ERROR_FLOPPY_WRONG_CYLINDER: i32 = 1123;
pub const ERROR_FORMS_AUTH_REQUIRED: i32 = 224;
pub const ERROR_FOUND_OUT_OF_SCOPE: i32 = 601;
pub const ERROR_FSFILTER_OP_COMPLETED_SUCCESSFULLY: i32 = 762;
pub const ERROR_FS_DRIVER_REQUIRED: i32 = 588;
pub const ERROR_FS_METADATA_INCONSISTENT: i32 = 510;
pub const ERROR_FT_DI_SCAN_REQUIRED: i32 = 339;
pub const ERROR_FT_READ_FAILURE: i32 = 415;
pub const ERROR_FT_READ_FROM_COPY_FAILURE: i32 = 818;
pub const ERROR_FT_READ_RECOVERY_FROM_BACKUP: i32 = 704;
pub const ERROR_FT_WRITE_FAILURE: i32 = 338;
pub const ERROR_FT_WRITE_RECOVERY: i32 = 705;
pub const ERROR_FULLSCREEN_MODE: i32 = 1007;
pub const ERROR_FUNCTION_FAILED: i32 = 1627;
pub const ERROR_FUNCTION_NOT_CALLED: i32 = 1626;
pub const ERROR_GDI_HANDLE_LEAK: i32 = 373;
pub const ERROR_GENERIC_NOT_MAPPED: i32 = 1360;
pub const ERROR_GEN_FAILURE: i32 = 31;
pub const ERROR_GLOBAL_ONLY_HOOK: i32 = 1429;
pub const ERROR_GRACEFUL_DISCONNECT: i32 = 1226;
pub const ERROR_GROUP_EXISTS: i32 = 1318;
pub const ERROR_GUID_SUBSTITUTION_MADE: i32 = 680;
pub const ERROR_HANDLES_CLOSED: i32 = 676;
pub const ERROR_HANDLE_DISK_FULL: i32 = 39;
pub const ERROR_HANDLE_EOF: i32 = 38;
pub const ERROR_HANDLE_REVOKED: i32 = 811;
pub const ERROR_HAS_SYSTEM_CRITICAL_FILES: i32 = 488;
pub const ERROR_HIBERNATED: i32 = 726;
pub const ERROR_HIBERNATION_FAILURE: i32 = 656;
pub const ERROR_HOOK_NEEDS_HMOD: i32 = 1428;
pub const ERROR_HOOK_NOT_INSTALLED: i32 = 1431;
pub const ERROR_HOOK_TYPE_NOT_ALLOWED: i32 = 1458;
pub const ERROR_HOST_DOWN: i32 = 1256;
pub const ERROR_HOST_UNREACHABLE: i32 = 1232;
pub const ERROR_HOTKEY_ALREADY_REGISTERED: i32 = 1409;
pub const ERROR_HOTKEY_NOT_REGISTERED: i32 = 1419;
pub const ERROR_HWNDS_HAVE_DIFF_PARENT: i32 = 1441;
pub const ERROR_ILLEGAL_CHARACTER: i32 = 582;
pub const ERROR_ILLEGAL_DLL_RELOCATION: i32 = 623;
pub const ERROR_ILLEGAL_ELEMENT_ADDRESS: i32 = 1162;
pub const ERROR_ILLEGAL_FLOAT_CONTEXT: i32 = 579;
pub const ERROR_ILL_FORMED_PASSWORD: i32 = 1324;
pub const ERROR_IMAGE_AT_DIFFERENT_BASE: i32 = 807;
pub const ERROR_IMAGE_MACHINE_TYPE_MISMATCH: i32 = 706;
pub const ERROR_IMAGE_MACHINE_TYPE_MISMATCH_EXE: i32 = 720;
pub const ERROR_IMAGE_NOT_AT_BASE: i32 = 700;
pub const ERROR_IMAGE_SUBSYSTEM_NOT_PRESENT: i32 = 308;
pub const ERROR_IMPLEMENTATION_LIMIT: i32 = 1292;
pub const ERROR_INCOMPATIBLE_SERVICE_PRIVILEGE: i32 = 1297;
pub const ERROR_INCOMPATIBLE_SERVICE_SID_TYPE: i32 = 1290;
pub const ERROR_INCOMPATIBLE_WITH_GLOBAL_SHORT_NAME_REGISTRY_SETTING: i32 = 304;
pub const ERROR_INCORRECT_ACCOUNT_TYPE: i32 = 8646;
pub const ERROR_INCORRECT_ADDRESS: i32 = 1241;
pub const ERROR_INCORRECT_SIZE: i32 = 1462;
pub const ERROR_INDEX_ABSENT: i32 = 1611;
pub const ERROR_INDEX_OUT_OF_BOUNDS: i32 = 474;
pub const ERROR_INFLOOP_IN_RELOC_CHAIN: i32 = 202;
pub const ERROR_INSTALL_ALREADY_RUNNING: i32 = 1618;
pub const ERROR_INSTALL_FAILURE: i32 = 1603;
pub const ERROR_INSTALL_LANGUAGE_UNSUPPORTED: i32 = 1623;
pub const ERROR_INSTALL_LOG_FAILURE: i32 = 1622;
pub const ERROR_INSTALL_NOTUSED: i32 = 1634;
pub const ERROR_INSTALL_PACKAGE_INVALID: i32 = 1620;
pub const ERROR_INSTALL_PACKAGE_OPEN_FAILED: i32 = 1619;
pub const ERROR_INSTALL_PACKAGE_REJECTED: i32 = 1625;
pub const ERROR_INSTALL_PACKAGE_VERSION: i32 = 1613;
pub const ERROR_INSTALL_PLATFORM_UNSUPPORTED: i32 = 1633;
pub const ERROR_INSTALL_REJECTED: i32 = 1654;
pub const ERROR_INSTALL_REMOTE_DISALLOWED: i32 = 1640;
pub const ERROR_INSTALL_REMOTE_PROHIBITED: i32 = 1645;
pub const ERROR_INSTALL_SERVICE_FAILURE: i32 = 1601;
pub const ERROR_INSTALL_SERVICE_SAFEBOOT: i32 = 1652;
pub const ERROR_INSTALL_SOURCE_ABSENT: i32 = 1612;
pub const ERROR_INSTALL_SUSPEND: i32 = 1604;
pub const ERROR_INSTALL_TEMP_UNWRITABLE: i32 = 1632;
pub const ERROR_INSTALL_TRANSFORM_FAILURE: i32 = 1624;
pub const ERROR_INSTALL_TRANSFORM_REJECTED: i32 = 1644;
pub const ERROR_INSTALL_UI_FAILURE: i32 = 1621;
pub const ERROR_INSTALL_USEREXIT: i32 = 1602;
pub const ERROR_INSTRUCTION_MISALIGNMENT: i32 = 549;
pub const ERROR_INSUFFICIENT_BUFFER: i32 = 122;
pub const ERROR_INSUFFICIENT_LOGON_INFO: i32 = 608;
pub const ERROR_INSUFFICIENT_POWER: i32 = 639;
pub const ERROR_INSUFFICIENT_RESOURCE_FOR_SPECIFIED_SHARED_SECTION_SIZE: i32 = 781;
pub const ERROR_INSUFFICIENT_VIRTUAL_ADDR_RESOURCES: i32 = 473;
pub const ERROR_INTERMIXED_KERNEL_EA_OPERATION: i32 = 324;
pub const ERROR_INTERNAL_DB_CORRUPTION: i32 = 1358;
pub const ERROR_INTERNAL_DB_ERROR: i32 = 1383;
pub const ERROR_INTERNAL_ERROR: i32 = 1359;
pub const ERROR_INTERRUPT_STILL_CONNECTED: i32 = 764;
pub const ERROR_INTERRUPT_VECTOR_ALREADY_CONNECTED: i32 = 763;
pub const ERROR_INVALID_ACCEL_HANDLE: i32 = 1403;
pub const ERROR_INVALID_ACCESS: i32 = 12;
pub const ERROR_INVALID_ACCOUNT_NAME: i32 = 1315;
pub const ERROR_INVALID_ACE_CONDITION: i32 = 805;
pub const ERROR_INVALID_ACL: i32 = 1336;
pub const ERROR_INVALID_ADDRESS: i32 = 487;
pub const ERROR_INVALID_AT_INTERRUPT_TIME: i32 = 104;
pub const ERROR_INVALID_BLOCK: i32 = 9;
pub const ERROR_INVALID_BLOCK_LENGTH: i32 = 1106;
pub const ERROR_INVALID_CAP: i32 = 320;
pub const ERROR_INVALID_CATEGORY: i32 = 117;
pub const ERROR_INVALID_COMBOBOX_MESSAGE: i32 = 1422;
pub const ERROR_INVALID_COMMAND_LINE: i32 = 1639;
pub const ERROR_INVALID_COMPUTERNAME: i32 = 1210;
pub const ERROR_INVALID_CRUNTIME_PARAMETER: i32 = 1288;
pub const ERROR_INVALID_CURSOR_HANDLE: i32 = 1402;
pub const ERROR_INVALID_DATA: i32 = 13;
pub const ERROR_INVALID_DATATYPE: i32 = 1804;
pub const ERROR_INVALID_DEVICE_OBJECT_PARAMETER: i32 = 650;
pub const ERROR_INVALID_DLL: i32 = 1154;
pub const ERROR_INVALID_DOMAINNAME: i32 = 1212;
pub const ERROR_INVALID_DOMAIN_ROLE: i32 = 1354;
pub const ERROR_INVALID_DOMAIN_STATE: i32 = 1353;
pub const ERROR_INVALID_DRIVE: i32 = 15;
pub const ERROR_INVALID_DWP_HANDLE: i32 = 1405;
pub const ERROR_INVALID_EA_HANDLE: i32 = 278;
pub const ERROR_INVALID_EA_NAME: i32 = 254;
pub const ERROR_INVALID_EDIT_HEIGHT: i32 = 1424;
pub const ERROR_INVALID_ENVIRONMENT: i32 = 1805;
pub const ERROR_INVALID_EVENTNAME: i32 = 1211;
pub const ERROR_INVALID_EVENT_COUNT: i32 = 151;
pub const ERROR_INVALID_EXCEPTION_HANDLER: i32 = 310;
pub const ERROR_INVALID_EXE_SIGNATURE: i32 = 191;
pub const ERROR_INVALID_FIELD: i32 = 1616;
pub const ERROR_INVALID_FIELD_IN_PARAMETER_LIST: i32 = 328;
pub const ERROR_INVALID_FILTER_PROC: i32 = 1427;
pub const ERROR_INVALID_FLAGS: i32 = 1004;
pub const ERROR_INVALID_FLAG_NUMBER: i32 = 186;
pub const ERROR_INVALID_FORM_NAME: i32 = 1902;
pub const ERROR_INVALID_FORM_SIZE: i32 = 1903;
pub const ERROR_INVALID_FUNCTION: i32 = 1;
pub const ERROR_INVALID_GROUPNAME: i32 = 1209;
pub const ERROR_INVALID_GROUP_ATTRIBUTES: i32 = 1345;
pub const ERROR_INVALID_GW_COMMAND: i32 = 1443;
pub const ERROR_INVALID_HANDLE: i32 = 6;
pub const ERROR_INVALID_HANDLE_STATE: i32 = 1609;
pub const ERROR_INVALID_HOOK_FILTER: i32 = 1426;
pub const ERROR_INVALID_HOOK_HANDLE: i32 = 1404;
pub const ERROR_INVALID_HW_PROFILE: i32 = 619;
pub const ERROR_INVALID_ICON_HANDLE: i32 = 1414;
pub const ERROR_INVALID_ID_AUTHORITY: i32 = 1343;
pub const ERROR_INVALID_IMAGE_HASH: i32 = 577;
pub const ERROR_INVALID_IMPORT_OF_NON_DLL: i32 = 1276;
pub const ERROR_INVALID_INDEX: i32 = 1413;
pub const ERROR_INVALID_KERNEL_INFO_VERSION: i32 = 340;
pub const ERROR_INVALID_KEYBOARD_HANDLE: i32 = 1457;
pub const ERROR_INVALID_LABEL: i32 = 1299;
pub const ERROR_INVALID_LB_MESSAGE: i32 = 1432;
pub const ERROR_INVALID_LDT_DESCRIPTOR: i32 = 564;
pub const ERROR_INVALID_LDT_OFFSET: i32 = 563;
pub const ERROR_INVALID_LDT_SIZE: i32 = 561;
pub const ERROR_INVALID_LEVEL: i32 = 124;
pub const ERROR_INVALID_LIST_FORMAT: i32 = 153;
pub const ERROR_INVALID_LOCK_RANGE: i32 = 307;
pub const ERROR_INVALID_LOGON_HOURS: i32 = 1328;
pub const ERROR_INVALID_LOGON_TYPE: i32 = 1367;
pub const ERROR_INVALID_MEMBER: i32 = 1388;
pub const ERROR_INVALID_MENU_HANDLE: i32 = 1401;
pub const ERROR_INVALID_MESSAGE: i32 = 1002;
pub const ERROR_INVALID_MESSAGEDEST: i32 = 1218;
pub const ERROR_INVALID_MESSAGENAME: i32 = 1217;
pub const ERROR_INVALID_MINALLOCSIZE: i32 = 195;
pub const ERROR_INVALID_MODULETYPE: i32 = 190;
pub const ERROR_INVALID_MONITOR_HANDLE: i32 = 1461;
pub const ERROR_INVALID_MSGBOX_STYLE: i32 = 1438;
pub const ERROR_INVALID_NAME: i32 = 123;
pub const ERROR_INVALID_NETNAME: i32 = 1214;
pub const ERROR_INVALID_OPLOCK_PROTOCOL: i32 = 301;
pub const ERROR_INVALID_ORDINAL: i32 = 182;
pub const ERROR_INVALID_OWNER: i32 = 1307;
pub const ERROR_INVALID_PACKAGE_SID_LENGTH: i32 = 4253;
pub const ERROR_INVALID_PARAMETER: i32 = 87;
pub const ERROR_INVALID_PASSWORD: i32 = 86;
pub const ERROR_INVALID_PASSWORDNAME: i32 = 1216;
pub const ERROR_INVALID_PATCH_XML: i32 = 1650;
pub const ERROR_INVALID_PEP_INFO_VERSION: i32 = 341;
pub const ERROR_INVALID_PLUGPLAY_DEVICE_PATH: i32 = 620;
pub const ERROR_INVALID_PORT_ATTRIBUTES: i32 = 545;
pub const ERROR_INVALID_PRIMARY_GROUP: i32 = 1308;
pub const ERROR_INVALID_PRINTER_COMMAND: i32 = 1803;
pub const ERROR_INVALID_PRINTER_NAME: i32 = 1801;
pub const ERROR_INVALID_PRINTER_STATE: i32 = 1906;
pub const ERROR_INVALID_PRIORITY: i32 = 1800;
pub const ERROR_INVALID_QUOTA_LOWER: i32 = 547;
pub const ERROR_INVALID_REPARSE_DATA: i32 = 4392;
pub const ERROR_INVALID_SCROLLBAR_RANGE: i32 = 1448;
pub const ERROR_INVALID_SECURITY_DESCR: i32 = 1338;
pub const ERROR_INVALID_SEGDPL: i32 = 198;
pub const ERROR_INVALID_SEGMENT_NUMBER: i32 = 180;
pub const ERROR_INVALID_SEPARATOR_FILE: i32 = 1799;
pub const ERROR_INVALID_SERVER_STATE: i32 = 1352;
pub const ERROR_INVALID_SERVICENAME: i32 = 1213;
pub const ERROR_INVALID_SERVICE_ACCOUNT: i32 = 1057;
pub const ERROR_INVALID_SERVICE_CONTROL: i32 = 1052;
pub const ERROR_INVALID_SERVICE_LOCK: i32 = 1071;
pub const ERROR_INVALID_SHARENAME: i32 = 1215;
pub const ERROR_INVALID_SHOWWIN_COMMAND: i32 = 1449;
pub const ERROR_INVALID_SID: i32 = 1337;
pub const ERROR_INVALID_SIGNAL_NUMBER: i32 = 209;
pub const ERROR_INVALID_SPI_VALUE: i32 = 1439;
pub const ERROR_INVALID_STACKSEG: i32 = 189;
pub const ERROR_INVALID_STARTING_CODESEG: i32 = 188;
pub const ERROR_INVALID_SUB_AUTHORITY: i32 = 1335;
pub const ERROR_INVALID_TABLE: i32 = 1628;
pub const ERROR_INVALID_TARGET_HANDLE: i32 = 114;
pub const ERROR_INVALID_TASK_INDEX: i32 = 1551;
pub const ERROR_INVALID_TASK_NAME: i32 = 1550;
pub const ERROR_INVALID_THREAD_ID: i32 = 1444;
pub const ERROR_INVALID_TIME: i32 = 1901;
pub const ERROR_INVALID_TOKEN: i32 = 315;
pub const ERROR_INVALID_UNWIND_TARGET: i32 = 544;
pub const ERROR_INVALID_USER_BUFFER: i32 = 1784;
pub const ERROR_INVALID_USER_PRINCIPAL_NAME: i32 = 8636;
pub const ERROR_INVALID_VARIANT: i32 = 604;
pub const ERROR_INVALID_VERIFY_SWITCH: i32 = 118;
pub const ERROR_INVALID_WINDOW_HANDLE: i32 = 1400;
pub const ERROR_INVALID_WORKSTATION: i32 = 1329;
pub const ERROR_IOPL_NOT_ENABLED: i32 = 197;
pub const ERROR_IO_DEVICE: i32 = 1117;
pub const ERROR_IO_INCOMPLETE: i32 = 996;
pub const ERROR_IO_PENDING: i32 = 997;
pub const ERROR_IO_PRIVILEGE_FAILED: i32 = 571;
pub const ERROR_IO_REISSUE_AS_CACHED: i32 = 3950;
pub const ERROR_IPSEC_IKE_TIMED_OUT: i32 = 13805;
pub const ERROR_IP_ADDRESS_CONFLICT1: i32 = 611;
pub const ERROR_IP_ADDRESS_CONFLICT2: i32 = 612;
pub const ERROR_IRQ_BUSY: i32 = 1119;
pub const ERROR_IS_JOINED: i32 = 134;
pub const ERROR_IS_JOIN_PATH: i32 = 147;
pub const ERROR_IS_JOIN_TARGET: i32 = 133;
pub const ERROR_IS_SUBSTED: i32 = 135;
pub const ERROR_IS_SUBST_PATH: i32 = 146;
pub const ERROR_IS_SUBST_TARGET: i32 = 149;
pub const ERROR_ITERATED_DATA_EXCEEDS_64k: i32 = 194;
pub const ERROR_JOB_NO_CONTAINER: i32 = 1505;
pub const ERROR_JOIN_TO_JOIN: i32 = 138;
pub const ERROR_JOIN_TO_SUBST: i32 = 140;
pub const ERROR_JOURNAL_DELETE_IN_PROGRESS: i32 = 1178;
pub const ERROR_JOURNAL_ENTRY_DELETED: i32 = 1181;
pub const ERROR_JOURNAL_HOOK_SET: i32 = 1430;
pub const ERROR_JOURNAL_NOT_ACTIVE: i32 = 1179;
pub const ERROR_KERNEL_APC: i32 = 738;
pub const ERROR_KEY_DELETED: i32 = 1018;
pub const ERROR_KEY_HAS_CHILDREN: i32 = 1020;
pub const ERROR_KM_DRIVER_BLOCKED: i32 = 1930;
pub const ERROR_LABEL_TOO_LONG: i32 = 154;
pub const ERROR_LAST_ADMIN: i32 = 1322;
pub const ERROR_LB_WITHOUT_TABSTOPS: i32 = 1434;
pub const ERROR_LICENSE_QUOTA_EXCEEDED: i32 = 1395;
pub const ERROR_LINUX_SUBSYSTEM_NOT_PRESENT: i32 = 414;
pub const ERROR_LINUX_SUBSYSTEM_UPDATE_REQUIRED: i32 = 444;
pub const ERROR_LISTBOX_ID_NOT_FOUND: i32 = 1416;
pub const ERROR_LM_CROSS_ENCRYPTION_REQUIRED: i32 = 1390;
pub const ERROR_LOCAL_POLICY_MODIFICATION_NOT_SUPPORTED: i32 = 8653;
pub const ERROR_LOCAL_USER_SESSION_KEY: i32 = 1303;
pub const ERROR_LOCKED: i32 = 212;
pub const ERROR_LOCK_FAILED: i32 = 167;
pub const ERROR_LOCK_VIOLATION: i32 = 33;
pub const ERROR_LOGIN_TIME_RESTRICTION: i32 = 1239;
pub const ERROR_LOGIN_WKSTA_RESTRICTION: i32 = 1240;
pub const ERROR_LOGON_FAILURE: i32 = 1326;
pub const ERROR_LOGON_NOT_GRANTED: i32 = 1380;
pub const ERROR_LOGON_SERVER_CONFLICT: i32 = 568;
pub const ERROR_LOGON_SESSION_COLLISION: i32 = 1366;
pub const ERROR_LOGON_SESSION_EXISTS: i32 = 1363;
pub const ERROR_LOGON_TYPE_NOT_GRANTED: i32 = 1385;
pub const ERROR_LOG_FILE_FULL: i32 = 1502;
pub const ERROR_LOG_HARD_ERROR: i32 = 718;
pub const ERROR_LONGJUMP: i32 = 682;
pub const ERROR_LOST_MODE_LOGON_RESTRICTION: i32 = 1939;
pub const ERROR_LOST_WRITEBEHIND_DATA: i32 = 596;
pub const ERROR_LOST_WRITEBEHIND_DATA_LOCAL_DISK_ERROR: i32 = 790;
pub const ERROR_LOST_WRITEBEHIND_DATA_NETWORK_DISCONNECTED: i32 = 788;
pub const ERROR_LOST_WRITEBEHIND_DATA_NETWORK_SERVER_ERROR: i32 = 789;
pub const ERROR_LUIDS_EXHAUSTED: i32 = 1334;
pub const ERROR_MACHINE_LOCKED: i32 = 1271;
pub const ERROR_MAGAZINE_NOT_PRESENT: i32 = 1163;
pub const ERROR_MAPPED_ALIGNMENT: i32 = 1132;
pub const ERROR_MARKED_TO_DISALLOW_WRITES: i32 = 348;
pub const ERROR_MARSHALL_OVERFLOW: i32 = 603;
pub const ERROR_MAX_SESSIONS_REACHED: i32 = 353;
pub const ERROR_MAX_THRDS_REACHED: i32 = 164;
pub const ERROR_MCA_EXCEPTION: i32 = 784;
pub const ERROR_MCA_OCCURED: i32 = 651;
pub const ERROR_MEDIA_CHANGED: i32 = 1110;
pub const ERROR_MEDIA_CHECK: i32 = 679;
pub const ERROR_MEMBERS_PRIMARY_GROUP: i32 = 1374;
pub const ERROR_MEMBER_IN_ALIAS: i32 = 1378;
pub const ERROR_MEMBER_IN_GROUP: i32 = 1320;
pub const ERROR_MEMBER_NOT_IN_ALIAS: i32 = 1377;
pub const ERROR_MEMBER_NOT_IN_GROUP: i32 = 1321;
pub const ERROR_MEMORY_HARDWARE: i32 = 779;
pub const ERROR_MENU_ITEM_NOT_FOUND: i32 = 1456;
pub const ERROR_MESSAGE_SYNC_ONLY: i32 = 1159;
pub const ERROR_META_EXPANSION_TOO_LONG: i32 = 208;
pub const ERROR_MISSING_SYSTEMFILE: i32 = 573;
pub const ERROR_MOD_NOT_FOUND: i32 = 126;
pub const ERROR_MORE_DATA: i32 = 234;
pub const ERROR_MORE_WRITES: i32 = 1120;
pub const ERROR_MOUNT_POINT_NOT_RESOLVED: i32 = 649;
pub const ERROR_MP_PROCESSOR_MISMATCH: i32 = 725;
pub const ERROR_MR_MID_NOT_FOUND: i32 = 317;
pub const ERROR_MULTIPLE_FAULT_VIOLATION: i32 = 640;
pub const ERROR_MUTANT_LIMIT_EXCEEDED: i32 = 587;
pub const ERROR_MUTUAL_AUTH_FAILED: i32 = 1397;
pub const ERROR_NEGATIVE_SEEK: i32 = 131;
pub const ERROR_NESTING_NOT_ALLOWED: i32 = 215;
pub const ERROR_NETLOGON_NOT_STARTED: i32 = 1792;
pub const ERROR_NETNAME_DELETED: i32 = 64;
pub const ERROR_NETWORK_ACCESS_DENIED: i32 = 65;
pub const ERROR_NETWORK_ACCESS_DENIED_EDP: i32 = 354;
pub const ERROR_NETWORK_BUSY: i32 = 54;
pub const ERROR_NETWORK_UNREACHABLE: i32 = 1231;
pub const ERROR_NET_OPEN_FAILED: i32 = 570;
pub const ERROR_NET_WRITE_FAULT: i32 = 88;
pub const ERROR_NOACCESS: i32 = 998;
pub const ERROR_NOINTERFACE: i32 = 632;
pub const ERROR_NOLOGON_INTERDOMAIN_TRUST_ACCOUNT: i32 = 1807;
pub const ERROR_NOLOGON_SERVER_TRUST_ACCOUNT: i32 = 1809;
pub const ERROR_NOLOGON_WORKSTATION_TRUST_ACCOUNT: i32 = 1808;
pub const ERROR_NONE_MAPPED: i32 = 1332;
pub const ERROR_NONPAGED_SYSTEM_RESOURCES: i32 = 1451;
pub const ERROR_NON_ACCOUNT_SID: i32 = 1257;
pub const ERROR_NON_DOMAIN_SID: i32 = 1258;
pub const ERROR_NON_MDICHILD_WINDOW: i32 = 1445;
pub const ERROR_NOTHING_TO_TERMINATE: i32 = 758;
pub const ERROR_NOTIFICATION_GUID_ALREADY_DEFINED: i32 = 309;
pub const ERROR_NOTIFY_CLEANUP: i32 = 745;
pub const ERROR_NOTIFY_ENUM_DIR: i32 = 1022;
pub const ERROR_NOT_ALLOWED_ON_SYSTEM_FILE: i32 = 313;
pub const ERROR_NOT_ALL_ASSIGNED: i32 = 1300;
pub const ERROR_NOT_APPCONTAINER: i32 = 4250;
pub const ERROR_NOT_AUTHENTICATED: i32 = 1244;
pub const ERROR_NOT_A_CLOUD_FILE: i32 = 376;
pub const ERROR_NOT_A_CLOUD_SYNC_ROOT: i32 = 405;
pub const ERROR_NOT_A_DAX_VOLUME: i32 = 420;
pub const ERROR_NOT_A_REPARSE_POINT: i32 = 4390;
pub const ERROR_NOT_CAPABLE: i32 = 775;
pub const ERROR_NOT_CHILD_WINDOW: i32 = 1442;
pub const ERROR_NOT_CONNECTED: i32 = 2250;
pub const ERROR_NOT_CONTAINER: i32 = 1207;
pub const ERROR_NOT_DAX_MAPPABLE: i32 = 421;
pub const ERROR_NOT_DOS_DISK: i32 = 26;
pub const ERROR_NOT_ENOUGH_MEMORY: i32 = 8;
pub const ERROR_NOT_ENOUGH_QUOTA: i32 = 1816;
pub const ERROR_NOT_ENOUGH_SERVER_MEMORY: i32 = 1130;
pub const ERROR_NOT_EXPORT_FORMAT: i32 = 6008;
pub const ERROR_NOT_FOUND: i32 = 1168;
pub const ERROR_NOT_GUI_PROCESS: i32 = 1471;
pub const ERROR_NOT_JOINED: i32 = 136;
pub const ERROR_NOT_LOCKED: i32 = 158;
pub const ERROR_NOT_LOGGED_ON: i32 = 1245;
pub const ERROR_NOT_LOGON_PROCESS: i32 = 1362;
pub const ERROR_NOT_OWNER: i32 = 288;
pub const ERROR_NOT_READY: i32 = 21;
pub const ERROR_NOT_READ_FROM_COPY: i32 = 337;
pub const ERROR_NOT_REDUNDANT_STORAGE: i32 = 333;
pub const ERROR_NOT_REGISTRY_FILE: i32 = 1017;
pub const ERROR_NOT_SAFEBOOT_SERVICE: i32 = 1084;
pub const ERROR_NOT_SAFE_MODE_DRIVER: i32 = 646;
pub const ERROR_NOT_SAME_DEVICE: i32 = 17;
pub const ERROR_NOT_SAME_OBJECT: i32 = 1656;
pub const ERROR_NOT_SUBSTED: i32 = 137;
pub const ERROR_NOT_SUPPORTED: i32 = 50;
pub const ERROR_NOT_SUPPORTED_IN_APPCONTAINER: i32 = 4252;
pub const ERROR_NOT_SUPPORTED_ON_DAX: i32 = 360;
pub const ERROR_NOT_SUPPORTED_ON_SBS: i32 = 1254;
pub const ERROR_NOT_SUPPORTED_ON_STANDARD_SERVER: i32 = 8584;
pub const ERROR_NOT_SUPPORTED_WITH_AUDITING: i32 = 499;
pub const ERROR_NOT_SUPPORTED_WITH_BTT: i32 = 429;
pub const ERROR_NOT_SUPPORTED_WITH_BYPASSIO: i32 = 493;
pub const ERROR_NOT_SUPPORTED_WITH_CACHED_HANDLE: i32 = 509;
pub const ERROR_NOT_SUPPORTED_WITH_COMPRESSION: i32 = 496;
pub const ERROR_NOT_SUPPORTED_WITH_DEDUPLICATION: i32 = 498;
pub const ERROR_NOT_SUPPORTED_WITH_ENCRYPTION: i32 = 495;
pub const ERROR_NOT_SUPPORTED_WITH_MONITORING: i32 = 503;
pub const ERROR_NOT_SUPPORTED_WITH_REPLICATION: i32 = 497;
pub const ERROR_NOT_SUPPORTED_WITH_SNAPSHOT: i32 = 504;
pub const ERROR_NOT_SUPPORTED_WITH_VIRTUALIZATION: i32 = 505;
pub const ERROR_NOT_TINY_STREAM: i32 = 598;
pub const ERROR_NO_ACE_CONDITION: i32 = 804;
pub const ERROR_NO_ASSOCIATION: i32 = 1155;
pub const ERROR_NO_BYPASSIO_DRIVER_SUPPORT: i32 = 494;
pub const ERROR_NO_CALLBACK_ACTIVE: i32 = 614;
pub const ERROR_NO_DATA: i32 = 232;
pub const ERROR_NO_DATA_DETECTED: i32 = 1104;
pub const ERROR_NO_EFS: i32 = 6004;
pub const ERROR_NO_EVENT_PAIR: i32 = 580;
pub const ERROR_NO_GUID_TRANSLATION: i32 = 560;
pub const ERROR_NO_IMPERSONATION_TOKEN: i32 = 1309;
pub const ERROR_NO_INHERITANCE: i32 = 1391;
pub const ERROR_NO_LOGON_SERVERS: i32 = 1311;
pub const ERROR_NO_LOG_SPACE: i32 = 1019;
pub const ERROR_NO_MATCH: i32 = 1169;
pub const ERROR_NO_MEDIA_IN_DRIVE: i32 = 1112;
pub const ERROR_NO_MORE_DEVICES: i32 = 1248;
pub const ERROR_NO_MORE_FILES: i32 = 18;
pub const ERROR_NO_MORE_ITEMS: i32 = 259;
pub const ERROR_NO_MORE_MATCHES: i32 = 626;
pub const ERROR_NO_MORE_SEARCH_HANDLES: i32 = 113;
pub const ERROR_NO_MORE_USER_HANDLES: i32 = 1158;
pub const ERROR_NO_NETWORK: i32 = 1222;
pub const ERROR_NO_NET_OR_BAD_PATH: i32 = 1203;
pub const ERROR_NO_NVRAM_RESOURCES: i32 = 1470;
pub const ERROR_NO_PAGEFILE: i32 = 578;
pub const ERROR_NO_PHYSICALLY_ALIGNED_FREE_SPACE_FOUND: i32 = 408;
pub const ERROR_NO_PROC_SLOTS: i32 = 89;
pub const ERROR_NO_PROMOTION_ACTIVE: i32 = 8222;
pub const ERROR_NO_QUOTAS_FOR_ACCOUNT: i32 = 1302;
pub const ERROR_NO_RANGES_PROCESSED: i32 = 312;
pub const ERROR_NO_RECOVERY_POLICY: i32 = 6003;
pub const ERROR_NO_RECOVERY_PROGRAM: i32 = 1082;
pub const ERROR_NO_SCROLLBARS: i32 = 1447;
pub const ERROR_NO_SECRETS: i32 = 8620;
pub const ERROR_NO_SECURITY_ON_OBJECT: i32 = 1350;
pub const ERROR_NO_SHUTDOWN_IN_PROGRESS: i32 = 1116;
pub const ERROR_NO_SIGNAL_SENT: i32 = 205;
pub const ERROR_NO_SITENAME: i32 = 1919;
pub const ERROR_NO_SITE_SETTINGS_OBJECT: i32 = 8619;
pub const ERROR_NO_SPOOL_SPACE: i32 = 62;
pub const ERROR_NO_SUCH_ALIAS: i32 = 1376;
pub const ERROR_NO_SUCH_DEVICE: i32 = 433;
pub const ERROR_NO_SUCH_DOMAIN: i32 = 1355;
pub const ERROR_NO_SUCH_GROUP: i32 = 1319;
pub const ERROR_NO_SUCH_LOGON_SESSION: i32 = 1312;
pub const ERROR_NO_SUCH_MEMBER: i32 = 1387;
pub const ERROR_NO_SUCH_PACKAGE: i32 = 1364;
pub const ERROR_NO_SUCH_PRIVILEGE: i32 = 1313;
pub const ERROR_NO_SUCH_SITE: i32 = 1249;
pub const ERROR_NO_SUCH_USER: i32 = 1317;
pub const ERROR_NO_SYSTEM_MENU: i32 = 1437;
pub const ERROR_NO_SYSTEM_RESOURCES: i32 = 1450;
pub const ERROR_NO_TASK_QUEUE: i32 = 427;
pub const ERROR_NO_TOKEN: i32 = 1008;
pub const ERROR_NO_TRACKING_SERVICE: i32 = 1172;
pub const ERROR_NO_TRUST_LSA_SECRET: i32 = 1786;
pub const ERROR_NO_TRUST_SAM_ACCOUNT: i32 = 1787;
pub const ERROR_NO_UNICODE_TRANSLATION: i32 = 1113;
pub const ERROR_NO_USER_KEYS: i32 = 6006;
pub const ERROR_NO_USER_SESSION_KEY: i32 = 1394;
pub const ERROR_NO_VOLUME_ID: i32 = 1173;
pub const ERROR_NO_VOLUME_LABEL: i32 = 125;
pub const ERROR_NO_WILDCARD_CHARACTERS: i32 = 1417;
pub const ERROR_NO_WORK_DONE: i32 = 235;
pub const ERROR_NO_WRITABLE_DC_FOUND: i32 = 8621;
pub const ERROR_NO_YIELD_PERFORMED: i32 = 721;
pub const ERROR_NTLM_BLOCKED: i32 = 1937;
pub const ERROR_NT_CROSS_ENCRYPTION_REQUIRED: i32 = 1386;
pub const ERROR_NULL_LM_PASSWORD: i32 = 1304;
pub const ERROR_OBJECT_IS_IMMUTABLE: i32 = 4449;
pub const ERROR_OBJECT_NAME_EXISTS: i32 = 698;
pub const ERROR_OBJECT_NOT_EXTERNALLY_BACKED: i32 = 342;
pub const ERROR_OFFLOAD_READ_FILE_NOT_SUPPORTED: i32 = 4442;
pub const ERROR_OFFLOAD_READ_FLT_NOT_SUPPORTED: i32 = 4440;
pub const ERROR_OFFLOAD_WRITE_FILE_NOT_SUPPORTED: i32 = 4443;
pub const ERROR_OFFLOAD_WRITE_FLT_NOT_SUPPORTED: i32 = 4441;
pub const ERROR_OFFSET_ALIGNMENT_VIOLATION: i32 = 327;
pub const ERROR_OLD_WIN_VERSION: i32 = 1150;
pub const ERROR_ONLY_IF_CONNECTED: i32 = 1251;
pub const ERROR_OPEN_FAILED: i32 = 110;
pub const ERROR_OPEN_FILES: i32 = 2401;
pub const ERROR_OPERATION_ABORTED: i32 = 995;
pub const ERROR_OPERATION_IN_PROGRESS: i32 = 329;
pub const ERROR_OPLOCK_BREAK_IN_PROGRESS: i32 = 742;
pub const ERROR_OPLOCK_HANDLE_CLOSED: i32 = 803;
pub const ERROR_OPLOCK_NOT_GRANTED: i32 = 300;
pub const ERROR_OPLOCK_SWITCHED_TO_NEW_HANDLE: i32 = 800;
pub const ERROR_ORPHAN_NAME_EXHAUSTED: i32 = 799;
pub const ERROR_OUTOFMEMORY: i32 = 14;
pub const ERROR_OUT_OF_PAPER: i32 = 28;
pub const ERROR_OUT_OF_STRUCTURES: i32 = 84;
pub const ERROR_OVERRIDE_NOCHANGES: i32 = 1252;
pub const ERROR_PAGED_SYSTEM_RESOURCES: i32 = 1452;
pub const ERROR_PAGEFILE_CREATE_FAILED: i32 = 576;
pub const ERROR_PAGEFILE_NOT_SUPPORTED: i32 = 491;
pub const ERROR_PAGEFILE_QUOTA: i32 = 1454;
pub const ERROR_PAGEFILE_QUOTA_EXCEEDED: i32 = 567;
pub const ERROR_PAGE_FAULT_COPY_ON_WRITE: i32 = 749;
pub const ERROR_PAGE_FAULT_DEMAND_ZERO: i32 = 748;
pub const ERROR_PAGE_FAULT_GUARD_PAGE: i32 = 750;
pub const ERROR_PAGE_FAULT_PAGING_FILE: i32 = 751;
pub const ERROR_PAGE_FAULT_TRANSITION: i32 = 747;
pub const ERROR_PARAMETER_QUOTA_EXCEEDED: i32 = 1283;
pub const ERROR_PARTIAL_COPY: i32 = 299;
pub const ERROR_PARTITION_FAILURE: i32 = 1105;
pub const ERROR_PARTITION_TERMINATING: i32 = 1184;
pub const ERROR_PASSWORD_CHANGE_REQUIRED: i32 = 1938;
pub const ERROR_PASSWORD_EXPIRED: i32 = 1330;
pub const ERROR_PASSWORD_MUST_CHANGE: i32 = 1907;
pub const ERROR_PASSWORD_RESTRICTION: i32 = 1325;
pub const ERROR_PATCH_MANAGED_ADVERTISED_PRODUCT: i32 = 1651;
pub const ERROR_PATCH_NO_SEQUENCE: i32 = 1648;
pub const ERROR_PATCH_PACKAGE_INVALID: i32 = 1636;
pub const ERROR_PATCH_PACKAGE_OPEN_FAILED: i32 = 1635;
pub const ERROR_PATCH_PACKAGE_REJECTED: i32 = 1643;
pub const ERROR_PATCH_PACKAGE_UNSUPPORTED: i32 = 1637;
pub const ERROR_PATCH_REMOVAL_DISALLOWED: i32 = 1649;
pub const ERROR_PATCH_REMOVAL_UNSUPPORTED: i32 = 1646;
pub const ERROR_PATCH_TARGET_NOT_FOUND: i32 = 1642;
pub const ERROR_PATH_BUSY: i32 = 148;
pub const ERROR_PATH_NOT_FOUND: i32 = 3;
pub const ERROR_PER_USER_TRUST_QUOTA_EXCEEDED: i32 = 1932;
pub const ERROR_PIPE_BUSY: i32 = 231;
pub const ERROR_PIPE_CONNECTED: i32 = 535;
pub const ERROR_PIPE_LISTENING: i32 = 536;
pub const ERROR_PIPE_LOCAL: i32 = 229;
pub const ERROR_PIPE_NOT_CONNECTED: i32 = 233;
pub const ERROR_PKINIT_FAILURE: i32 = 1263;
pub const ERROR_PLUGPLAY_QUERY_VETOED: i32 = 683;
pub const ERROR_PNP_BAD_MPS_TABLE: i32 = 671;
pub const ERROR_PNP_INVALID_ID: i32 = 674;
pub const ERROR_PNP_IRQ_TRANSLATION_FAILED: i32 = 673;
pub const ERROR_PNP_QUERY_REMOVE_DEVICE_TIMEOUT: i32 = 480;
pub const ERROR_PNP_QUERY_REMOVE_RELATED_DEVICE_TIMEOUT: i32 = 481;
pub const ERROR_PNP_QUERY_REMOVE_UNRELATED_DEVICE_TIMEOUT: i32 = 482;
pub const ERROR_PNP_REBOOT_REQUIRED: i32 = 638;
pub const ERROR_PNP_RESTART_ENUMERATION: i32 = 636;
pub const ERROR_PNP_TRANSLATION_FAILED: i32 = 672;
pub const ERROR_POINT_NOT_FOUND: i32 = 1171;
pub const ERROR_POLICY_OBJECT_NOT_FOUND: i32 = 8219;
pub const ERROR_POLICY_ONLY_IN_DS: i32 = 8220;
pub const ERROR_POPUP_ALREADY_ACTIVE: i32 = 1446;
pub const ERROR_PORT_MESSAGE_TOO_LONG: i32 = 546;
pub const ERROR_PORT_NOT_SET: i32 = 642;
pub const ERROR_PORT_UNREACHABLE: i32 = 1234;
pub const ERROR_POSSIBLE_DEADLOCK: i32 = 1131;
pub const ERROR_POTENTIAL_FILE_FOUND: i32 = 1180;
pub const ERROR_PREDEFINED_HANDLE: i32 = 714;
pub const ERROR_PRIMARY_TRANSPORT_CONNECT_FAILED: i32 = 746;
pub const ERROR_PRINTER_ALREADY_EXISTS: i32 = 1802;
pub const ERROR_PRINTER_DELETED: i32 = 1905;
pub const ERROR_PRINTER_DRIVER_ALREADY_INSTALLED: i32 = 1795;
pub const ERROR_PRINTQ_FULL: i32 = 61;
pub const ERROR_PRINT_CANCELLED: i32 = 63;
pub const ERROR_PRIVATE_DIALOG_INDEX: i32 = 1415;
pub const ERROR_PRIVILEGE_NOT_HELD: i32 = 1314;
pub const ERROR_PROCESS_ABORTED: i32 = 1067;
pub const ERROR_PROCESS_IN_JOB: i32 = 760;
pub const ERROR_PROCESS_IS_PROTECTED: i32 = 1293;
pub const ERROR_PROCESS_MODE_ALREADY_BACKGROUND: i32 = 402;
pub const ERROR_PROCESS_MODE_NOT_BACKGROUND: i32 = 403;
pub const ERROR_PROCESS_NOT_IN_JOB: i32 = 759;
pub const ERROR_PROC_NOT_FOUND: i32 = 127;
pub const ERROR_PRODUCT_UNINSTALLED: i32 = 1614;
pub const ERROR_PRODUCT_VERSION: i32 = 1638;
pub const ERROR_PROFILING_AT_LIMIT: i32 = 553;
pub const ERROR_PROFILING_NOT_STARTED: i32 = 550;
pub const ERROR_PROFILING_NOT_STOPPED: i32 = 551;
pub const ERROR_PROMOTION_ACTIVE: i32 = 8221;
pub const ERROR_PROTOCOL_UNREACHABLE: i32 = 1233;
pub const ERROR_PWD_HISTORY_CONFLICT: i32 = 617;
pub const ERROR_PWD_TOO_LONG: i32 = 657;
pub const ERROR_PWD_TOO_RECENT: i32 = 616;
pub const ERROR_PWD_TOO_SHORT: i32 = 615;
pub const ERROR_QUOTA_ACTIVITY: i32 = 810;
pub const ERROR_QUOTA_LIST_INCONSISTENT: i32 = 621;
pub const ERROR_RANGE_LIST_CONFLICT: i32 = 627;
pub const ERROR_RANGE_NOT_FOUND: i32 = 644;
pub const ERROR_READ_FAULT: i32 = 30;
pub const ERROR_RECEIVE_EXPEDITED: i32 = 708;
pub const ERROR_RECEIVE_PARTIAL: i32 = 707;
pub const ERROR_RECEIVE_PARTIAL_EXPEDITED: i32 = 709;
pub const ERROR_RECOVERY_FAILURE: i32 = 1279;
pub const ERROR_REDIRECTOR_HAS_OPEN_HANDLES: i32 = 1794;
pub const ERROR_REDIR_PAUSED: i32 = 72;
pub const ERROR_REGISTRY_CORRUPT: i32 = 1015;
pub const ERROR_REGISTRY_HIVE_RECOVERED: i32 = 685;
pub const ERROR_REGISTRY_IO_FAILED: i32 = 1016;
pub const ERROR_REGISTRY_QUOTA_LIMIT: i32 = 613;
pub const ERROR_REGISTRY_RECOVERED: i32 = 1014;
pub const ERROR_REG_NAT_CONSUMPTION: i32 = 1261;
pub const ERROR_RELOC_CHAIN_XEEDS_SEGLIM: i32 = 201;
pub const ERROR_REMOTE_PRINT_CONNECTIONS_BLOCKED: i32 = 1936;
pub const ERROR_REMOTE_SESSION_LIMIT_EXCEEDED: i32 = 1220;
pub const ERROR_REMOTE_STORAGE_MEDIA_ERROR: i32 = 4352;
pub const ERROR_REMOTE_STORAGE_NOT_ACTIVE: i32 = 4351;
pub const ERROR_REM_NOT_LIST: i32 = 51;
pub const ERROR_REPARSE: i32 = 741;
pub const ERROR_REPARSE_ATTRIBUTE_CONFLICT: i32 = 4391;
pub const ERROR_REPARSE_OBJECT: i32 = 755;
pub const ERROR_REPARSE_POINT_ENCOUNTERED: i32 = 4395;
pub const ERROR_REPARSE_TAG_INVALID: i32 = 4393;
pub const ERROR_REPARSE_TAG_MISMATCH: i32 = 4394;
pub const ERROR_REPLY_MESSAGE_MISMATCH: i32 = 595;
pub const ERROR_REQUEST_ABORTED: i32 = 1235;
pub const ERROR_REQUEST_OUT_OF_SEQUENCE: i32 = 776;
pub const ERROR_REQUEST_PAUSED: i32 = 3050;
pub const ERROR_REQUIRES_INTERACTIVE_WINDOWSTATION: i32 = 1459;
pub const ERROR_REQ_NOT_ACCEP: i32 = 71;
pub const ERROR_RESIDENT_FILE_NOT_SUPPORTED: i32 = 334;
pub const ERROR_RESOURCE_CALL_TIMED_OUT: i32 = 5910;
pub const ERROR_RESOURCE_DATA_NOT_FOUND: i32 = 1812;
pub const ERROR_RESOURCE_LANG_NOT_FOUND: i32 = 1815;
pub const ERROR_RESOURCE_NAME_NOT_FOUND: i32 = 1814;
pub const ERROR_RESOURCE_REQUIREMENTS_CHANGED: i32 = 756;
pub const ERROR_RESOURCE_TYPE_NOT_FOUND: i32 = 1813;
pub const ERROR_RESTART_APPLICATION: i32 = 1467;
pub const ERROR_RESUME_HIBERNATION: i32 = 727;
pub const ERROR_RETRY: i32 = 1237;
pub const ERROR_RETURN_ADDRESS_HIJACK_ATTEMPT: i32 = 1662;
pub const ERROR_REVISION_MISMATCH: i32 = 1306;
pub const ERROR_RING2SEG_MUST_BE_MOVABLE: i32 = 200;
pub const ERROR_RING2_STACK_IN_USE: i32 = 207;
pub const ERROR_RMODE_APP: i32 = 1153;
pub const ERROR_ROWSNOTRELEASED: i32 = 772;
pub const ERROR_RUNLEVEL_SWITCH_AGENT_TIMEOUT: i32 = 15403;
pub const ERROR_RUNLEVEL_SWITCH_TIMEOUT: i32 = 15402;
pub const ERROR_RWRAW_ENCRYPTED_FILE_NOT_ENCRYPTED: i32 = 410;
pub const ERROR_RWRAW_ENCRYPTED_INVALID_EDATAINFO_FILEOFFSET: i32 = 411;
pub const ERROR_RWRAW_ENCRYPTED_INVALID_EDATAINFO_FILERANGE: i32 = 412;
pub const ERROR_RWRAW_ENCRYPTED_INVALID_EDATAINFO_PARAMETER: i32 = 413;
pub const ERROR_RXACT_COMMITTED: i32 = 744;
pub const ERROR_RXACT_COMMIT_FAILURE: i32 = 1370;
pub const ERROR_RXACT_COMMIT_NECESSARY: i32 = 678;
pub const ERROR_RXACT_INVALID_STATE: i32 = 1369;
pub const ERROR_RXACT_STATE_CREATED: i32 = 701;
pub const ERROR_SAME_DRIVE: i32 = 143;
pub const ERROR_SAM_INIT_FAILURE: i32 = 8541;
pub const ERROR_SCOPE_NOT_FOUND: i32 = 318;
pub const ERROR_SCREEN_ALREADY_LOCKED: i32 = 1440;
pub const ERROR_SCRUB_DATA_DISABLED: i32 = 332;
pub const ERROR_SECRET_TOO_LONG: i32 = 1382;
pub const ERROR_SECTION_DIRECT_MAP_ONLY: i32 = 819;
pub const ERROR_SECTOR_NOT_FOUND: i32 = 27;
pub const ERROR_SECURITY_DENIES_OPERATION: i32 = 447;
pub const ERROR_SECURITY_STREAM_IS_INCONSISTENT: i32 = 306;
pub const ERROR_SEEK: i32 = 25;
pub const ERROR_SEEK_ON_DEVICE: i32 = 132;
pub const ERROR_SEGMENT_NOTIFICATION: i32 = 702;
pub const ERROR_SEM_IS_SET: i32 = 102;
pub const ERROR_SEM_NOT_FOUND: i32 = 187;
pub const ERROR_SEM_OWNER_DIED: i32 = 105;
pub const ERROR_SEM_TIMEOUT: i32 = 121;
pub const ERROR_SEM_USER_LIMIT: i32 = 106;
pub const ERROR_SERIAL_NO_DEVICE: i32 = 1118;
pub const ERROR_SERVER_DISABLED: i32 = 1341;
pub const ERROR_SERVER_HAS_OPEN_HANDLES: i32 = 1811;
pub const ERROR_SERVER_NOT_DISABLED: i32 = 1342;
pub const ERROR_SERVER_SHUTDOWN_IN_PROGRESS: i32 = 1255;
pub const ERROR_SERVER_SID_MISMATCH: i32 = 628;
pub const ERROR_SERVER_TRANSPORT_CONFLICT: i32 = 816;
pub const ERROR_SERVICE_ALREADY_RUNNING: i32 = 1056;
pub const ERROR_SERVICE_CANNOT_ACCEPT_CTRL: i32 = 1061;
pub const ERROR_SERVICE_DATABASE_LOCKED: i32 = 1055;
pub const ERROR_SERVICE_DEPENDENCY_DELETED: i32 = 1075;
pub const ERROR_SERVICE_DEPENDENCY_FAIL: i32 = 1068;
pub const ERROR_SERVICE_DISABLED: i32 = 1058;
pub const ERROR_SERVICE_DOES_NOT_EXIST: i32 = 1060;
pub const ERROR_SERVICE_EXISTS: i32 = 1073;
pub const ERROR_SERVICE_LOGON_FAILED: i32 = 1069;
pub const ERROR_SERVICE_MARKED_FOR_DELETE: i32 = 1072;
pub const ERROR_SERVICE_NEVER_STARTED: i32 = 1077;
pub const ERROR_SERVICE_NOTIFICATION: i32 = 716;
pub const ERROR_SERVICE_NOTIFY_CLIENT_LAGGING: i32 = 1294;
pub const ERROR_SERVICE_NOT_ACTIVE: i32 = 1062;
pub const ERROR_SERVICE_NOT_FOUND: i32 = 1243;
pub const ERROR_SERVICE_NOT_IN_EXE: i32 = 1083;
pub const ERROR_SERVICE_NO_THREAD: i32 = 1054;
pub const ERROR_SERVICE_REQUEST_TIMEOUT: i32 = 1053;
pub const ERROR_SERVICE_SPECIFIC_ERROR: i32 = 1066;
pub const ERROR_SERVICE_START_HANG: i32 = 1070;
pub const ERROR_SESSION_CREDENTIAL_CONFLICT: i32 = 1219;
pub const ERROR_SESSION_KEY_TOO_SHORT: i32 = 501;
pub const ERROR_SETCOUNT_ON_BAD_LB: i32 = 1433;
pub const ERROR_SETMARK_DETECTED: i32 = 1103;
pub const ERROR_SET_CONTEXT_DENIED: i32 = 1660;
pub const ERROR_SET_NOT_FOUND: i32 = 1170;
pub const ERROR_SET_POWER_STATE_FAILED: i32 = 1141;
pub const ERROR_SET_POWER_STATE_VETOED: i32 = 1140;
pub const ERROR_SHARED_POLICY: i32 = 8218;
pub const ERROR_SHARING_BUFFER_EXCEEDED: i32 = 36;
pub const ERROR_SHARING_PAUSED: i32 = 70;
pub const ERROR_SHARING_VIOLATION: i32 = 32;
pub const ERROR_SHORT_NAMES_NOT_ENABLED_ON_VOLUME: i32 = 305;
pub const ERROR_SHUTDOWN_DISKS_NOT_IN_MAINTENANCE_MODE: i32 = 1192;
pub const ERROR_SHUTDOWN_IN_PROGRESS: i32 = 1115;
pub const ERROR_SHUTDOWN_IS_SCHEDULED: i32 = 1190;
pub const ERROR_SHUTDOWN_USERS_LOGGED_ON: i32 = 1191;
pub const ERROR_SIGNAL_PENDING: i32 = 162;
pub const ERROR_SIGNAL_REFUSED: i32 = 156;
pub const ERROR_SINGLE_INSTANCE_APP: i32 = 1152;
pub const ERROR_SMARTCARD_SUBSYSTEM_FAILURE: i32 = 1264;
pub const ERROR_SMB1_NOT_AVAILABLE: i32 = 384;
pub const ERROR_SMB_GUEST_LOGON_BLOCKED: i32 = 1272;
pub const ERROR_SMR_GARBAGE_COLLECTION_REQUIRED: i32 = 4445;
pub const ERROR_SOME_NOT_MAPPED: i32 = 1301;
pub const ERROR_SOURCE_ELEMENT_EMPTY: i32 = 1160;
pub const ERROR_SPARSE_FILE_NOT_SUPPORTED: i32 = 490;
pub const ERROR_SPECIAL_ACCOUNT: i32 = 1371;
pub const ERROR_SPECIAL_GROUP: i32 = 1372;
pub const ERROR_SPECIAL_USER: i32 = 1373;
pub const ERROR_SRC_SRV_DLL_LOAD_FAILED: i32 = 428;
pub const ERROR_STACK_BUFFER_OVERRUN: i32 = 1282;
pub const ERROR_STACK_OVERFLOW: i32 = 1001;
pub const ERROR_STACK_OVERFLOW_READ: i32 = 599;
pub const ERROR_STOPPED_ON_SYMLINK: i32 = 681;
pub const ERROR_STORAGE_LOST_DATA_PERSISTENCE: i32 = 368;
pub const ERROR_STORAGE_RESERVE_ALREADY_EXISTS: i32 = 418;
pub const ERROR_STORAGE_RESERVE_DOES_NOT_EXIST: i32 = 417;
pub const ERROR_STORAGE_RESERVE_ID_INVALID: i32 = 416;
pub const ERROR_STORAGE_RESERVE_NOT_EMPTY: i32 = 419;
pub const ERROR_STORAGE_STACK_ACCESS_DENIED: i32 = 472;
pub const ERROR_STORAGE_TOPOLOGY_ID_MISMATCH: i32 = 345;
pub const ERROR_STRICT_CFG_VIOLATION: i32 = 1657;
pub const ERROR_SUBST_TO_JOIN: i32 = 141;
pub const ERROR_SUBST_TO_SUBST: i32 = 139;
pub const ERROR_SUCCESS: i32 = 0;
pub const ERROR_SUCCESS_REBOOT_INITIATED: i32 = 1641;
pub const ERROR_SWAPERROR: i32 = 999;
pub const ERROR_SYMLINK_CLASS_DISABLED: i32 = 1463;
pub const ERROR_SYMLINK_NOT_SUPPORTED: i32 = 1464;
pub const ERROR_SYNCHRONIZATION_REQUIRED: i32 = 569;
pub const ERROR_SYNC_FOREGROUND_REFRESH_REQUIRED: i32 = 1274;
pub const ERROR_SYSTEM_HIVE_TOO_LARGE: i32 = 653;
pub const ERROR_SYSTEM_IMAGE_BAD_SIGNATURE: i32 = 637;
pub const ERROR_SYSTEM_POWERSTATE_COMPLEX_TRANSITION: i32 = 783;
pub const ERROR_SYSTEM_POWERSTATE_TRANSITION: i32 = 782;
pub const ERROR_SYSTEM_PROCESS_TERMINATED: i32 = 591;
pub const ERROR_SYSTEM_SHUTDOWN: i32 = 641;
pub const ERROR_SYSTEM_TRACE: i32 = 150;
pub const ERROR_THREAD_1_INACTIVE: i32 = 210;
pub const ERROR_THREAD_ALREADY_IN_TASK: i32 = 1552;
pub const ERROR_THREAD_MODE_ALREADY_BACKGROUND: i32 = 400;
pub const ERROR_THREAD_MODE_NOT_BACKGROUND: i32 = 401;
pub const ERROR_THREAD_NOT_IN_PROCESS: i32 = 566;
pub const ERROR_THREAD_WAS_SUSPENDED: i32 = 699;
pub const ERROR_TIMEOUT: i32 = 1460;
pub const ERROR_TIMER_NOT_CANCELED: i32 = 541;
pub const ERROR_TIMER_RESOLUTION_NOT_SET: i32 = 607;
pub const ERROR_TIMER_RESUME_IGNORED: i32 = 722;
pub const ERROR_TIME_SENSITIVE_THREAD: i32 = 422;
pub const ERROR_TIME_SKEW: i32 = 1398;
pub const ERROR_TLW_WITH_WSCHILD: i32 = 1406;
pub const ERROR_TOKEN_ALREADY_IN_USE: i32 = 1375;
pub const ERROR_TOO_MANY_CMDS: i32 = 56;
pub const ERROR_TOO_MANY_CONTEXT_IDS: i32 = 1384;
pub const ERROR_TOO_MANY_DESCRIPTORS: i32 = 331;
pub const ERROR_TOO_MANY_LINKS: i32 = 1142;
pub const ERROR_TOO_MANY_LUIDS_REQUESTED: i32 = 1333;
pub const ERROR_TOO_MANY_MODULES: i32 = 214;
pub const ERROR_TOO_MANY_MUXWAITERS: i32 = 152;
pub const ERROR_TOO_MANY_NAMES: i32 = 68;
pub const ERROR_TOO_MANY_OPEN_FILES: i32 = 4;
pub const ERROR_TOO_MANY_POSTS: i32 = 298;
pub const ERROR_TOO_MANY_SECRETS: i32 = 1381;
pub const ERROR_TOO_MANY_SEMAPHORES: i32 = 100;
pub const ERROR_TOO_MANY_SEM_REQUESTS: i32 = 103;
pub const ERROR_TOO_MANY_SESS: i32 = 69;
pub const ERROR_TOO_MANY_SIDS: i32 = 1389;
pub const ERROR_TOO_MANY_TCBS: i32 = 155;
pub const ERROR_TOO_MANY_THREADS: i32 = 565;
pub const ERROR_TRANSLATION_COMPLETE: i32 = 757;
pub const ERROR_TRUSTED_DOMAIN_FAILURE: i32 = 1788;
pub const ERROR_TRUSTED_RELATIONSHIP_FAILURE: i32 = 1789;
pub const ERROR_TRUST_FAILURE: i32 = 1790;
pub const ERROR_UNABLE_TO_LOCK_MEDIA: i32 = 1108;
pub const ERROR_UNABLE_TO_MOVE_REPLACEMENT: i32 = 1176;
pub const ERROR_UNABLE_TO_MOVE_REPLACEMENT_2: i32 = 1177;
pub const ERROR_UNABLE_TO_REMOVE_REPLACED: i32 = 1175;
pub const ERROR_UNABLE_TO_UNLOAD_MEDIA: i32 = 1109;
pub const ERROR_UNDEFINED_CHARACTER: i32 = 583;
pub const ERROR_UNDEFINED_SCOPE: i32 = 319;
pub const ERROR_UNEXPECTED_MM_CREATE_ERR: i32 = 556;
pub const ERROR_UNEXPECTED_MM_EXTEND_ERR: i32 = 558;
pub const ERROR_UNEXPECTED_MM_MAP_ERROR: i32 = 557;
pub const ERROR_UNEXPECTED_NTCACHEMANAGER_ERROR: i32 = 443;
pub const ERROR_UNEXP_NET_ERR: i32 = 59;
pub const ERROR_UNHANDLED_EXCEPTION: i32 = 574;
pub const ERROR_UNIDENTIFIED_ERROR: i32 = 1287;
pub const ERROR_UNKNOWN_COMPONENT: i32 = 1607;
pub const ERROR_UNKNOWN_FEATURE: i32 = 1606;
pub const ERROR_UNKNOWN_PATCH: i32 = 1647;
pub const ERROR_UNKNOWN_PORT: i32 = 1796;
pub const ERROR_UNKNOWN_PRINTER_DRIVER: i32 = 1797;
pub const ERROR_UNKNOWN_PRINTPROCESSOR: i32 = 1798;
pub const ERROR_UNKNOWN_PRODUCT: i32 = 1605;
pub const ERROR_UNKNOWN_PROPERTY: i32 = 1608;
pub const ERROR_UNKNOWN_REVISION: i32 = 1305;
pub const ERROR_UNRECOGNIZED_MEDIA: i32 = 1785;
pub const ERROR_UNRECOGNIZED_VOLUME: i32 = 1005;
pub const ERROR_UNSATISFIED_DEPENDENCIES: i32 = 441;
pub const ERROR_UNSUPPORTED_COMPRESSION: i32 = 618;
pub const ERROR_UNSUPPORTED_TYPE: i32 = 1630;
pub const ERROR_UNTRUSTED_MOUNT_POINT: i32 = 448;
pub const ERROR_UNWIND: i32 = 542;
pub const ERROR_UNWIND_CONSOLIDATE: i32 = 684;
pub const ERROR_USER_APC: i32 = 737;
pub const ERROR_USER_DELETE_TRUST_QUOTA_EXCEEDED: i32 = 1934;
pub const ERROR_USER_EXISTS: i32 = 1316;
pub const ERROR_USER_MAPPED_FILE: i32 = 1224;
pub const ERROR_USER_PROFILE_LOAD: i32 = 500;
pub const ERROR_VALIDATE_CONTINUE: i32 = 625;
pub const ERROR_VC_DISCONNECTED: i32 = 240;
pub const ERROR_VDM_DISALLOWED: i32 = 1286;
pub const ERROR_VDM_HARD_ERROR: i32 = 593;
pub const ERROR_VERIFIER_STOP: i32 = 537;
pub const ERROR_VERSION_PARSE_ERROR: i32 = 777;
pub const ERROR_VIRUS_DELETED: i32 = 226;
pub const ERROR_VIRUS_INFECTED: i32 = 225;
pub const ERROR_VOLSNAP_HIBERNATE_READY: i32 = 761;
pub const ERROR_VOLSNAP_PREPARE_HIBERNATE: i32 = 655;
pub const ERROR_VOLUME_MOUNTED: i32 = 743;
pub const ERROR_VOLUME_NOT_CLUSTER_ALIGNED: i32 = 407;
pub const ERROR_VOLUME_NOT_SIS_ENABLED: i32 = 4500;
pub const ERROR_VOLUME_NOT_SUPPORTED: i32 = 492;
pub const ERROR_VOLUME_NOT_SUPPORT_EFS: i32 = 6014;
pub const ERROR_VOLUME_WRITE_ACCESS_DENIED: i32 = 508;
pub const ERROR_WAIT_1: i32 = 731;
pub const ERROR_WAIT_2: i32 = 732;
pub const ERROR_WAIT_3: i32 = 733;
pub const ERROR_WAIT_63: i32 = 734;
pub const ERROR_WAIT_FOR_OPLOCK: i32 = 765;
pub const ERROR_WAIT_NO_CHILDREN: i32 = 128;
pub const ERROR_WAKE_SYSTEM: i32 = 730;
pub const ERROR_WAKE_SYSTEM_DEBUGGER: i32 = 675;
pub const ERROR_WAS_LOCKED: i32 = 717;
pub const ERROR_WAS_UNLOCKED: i32 = 715;
pub const ERROR_WEAK_WHFBKEY_BLOCKED: i32 = 8651;
pub const ERROR_WINDOW_NOT_COMBOBOX: i32 = 1423;
pub const ERROR_WINDOW_NOT_DIALOG: i32 = 1420;
pub const ERROR_WINDOW_OF_OTHER_THREAD: i32 = 1408;
pub const ERROR_WIP_ENCRYPTION_FAILED: i32 = 6023;
pub const ERROR_WOF_FILE_RESOURCE_TABLE_CORRUPT: i32 = 4448;
pub const ERROR_WOF_WIM_HEADER_CORRUPT: i32 = 4446;
pub const ERROR_WOF_WIM_RESOURCE_TABLE_CORRUPT: i32 = 4447;
pub const ERROR_WORKING_SET_QUOTA: i32 = 1453;
pub const ERROR_WOW_ASSERTION: i32 = 670;
pub const ERROR_WRITE_FAULT: i32 = 29;
pub const ERROR_WRITE_PROTECT: i32 = 19;
pub const ERROR_WRONG_COMPARTMENT: i32 = 1468;
pub const ERROR_WRONG_DISK: i32 = 34;
pub const ERROR_WRONG_EFS: i32 = 6005;
pub const ERROR_WRONG_PASSWORD: i32 = 1323;
pub const ERROR_WRONG_TARGET_NAME: i32 = 1396;
pub const ERROR_WX86_ERROR: i32 = 540;
pub const ERROR_WX86_WARNING: i32 = 539;
pub const ERROR_XMLDSIG_ERROR: i32 = 1466;
pub const ERROR_XML_PARSE_ERROR: i32 = 1465;
pub type EXCEPTION_DISPOSITION = i32;
pub const EXCEPTION_MAXIMUM_PARAMETERS: i32 = 15;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EXCEPTION_POINTERS {
    pub ExceptionRecord: PEXCEPTION_RECORD,
    pub ContextRecord: PCONTEXT,
}
impl Default for EXCEPTION_POINTERS {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EXCEPTION_RECORD {
    pub ExceptionCode: u32,
    pub ExceptionFlags: u32,
    pub ExceptionRecord: *mut Self,
    pub ExceptionAddress: *mut core::ffi::c_void,
    pub NumberParameters: u32,
    pub ExceptionInformation: [usize; 15],
}
impl Default for EXCEPTION_RECORD {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const EXCEPTION_STACK_OVERFLOW: i32 = -1073741571;
pub const EXTENDED_STARTUPINFO_PRESENT: i32 = 524288;
pub const E_NOTIMPL: HRESULT = 0x80004001_u32 as _;
pub const ExceptionCollidedUnwind: EXCEPTION_DISPOSITION = 3;
pub const ExceptionContinueExecution: EXCEPTION_DISPOSITION = 0;
pub const ExceptionContinueSearch: EXCEPTION_DISPOSITION = 1;
pub const ExceptionNestedException: EXCEPTION_DISPOSITION = 2;
pub const FACILITY_NT_BIT: i32 = 268435456;
pub const FALSE: i32 = 0;
pub type FARPROC = Option<unsafe extern "system" fn() -> isize>;
pub const FAST_FAIL_FATAL_APP_EXIT: i32 = 7;
pub type FD_SET = fd_set;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILETIME {
    pub dwLowDateTime: u32,
    pub dwHighDateTime: u32,
}
pub const FILE_ADD_FILE: i32 = 2;
pub const FILE_ADD_SUBDIRECTORY: i32 = 4;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_ALLOCATION_INFO {
    pub AllocationSize: i64,
}
pub const FILE_ALL_ACCESS: i32 = 2032127;
pub const FILE_APPEND_DATA: i32 = 4;
pub const FILE_ATTRIBUTE_ARCHIVE: i32 = 32;
pub const FILE_ATTRIBUTE_COMPRESSED: i32 = 2048;
pub const FILE_ATTRIBUTE_DEVICE: i32 = 64;
pub const FILE_ATTRIBUTE_DIRECTORY: i32 = 16;
pub const FILE_ATTRIBUTE_EA: i32 = 262144;
pub const FILE_ATTRIBUTE_ENCRYPTED: i32 = 16384;
pub const FILE_ATTRIBUTE_HIDDEN: i32 = 2;
pub const FILE_ATTRIBUTE_INTEGRITY_STREAM: i32 = 32768;
pub const FILE_ATTRIBUTE_NORMAL: i32 = 128;
pub const FILE_ATTRIBUTE_NOT_CONTENT_INDEXED: i32 = 8192;
pub const FILE_ATTRIBUTE_NO_SCRUB_DATA: i32 = 131072;
pub const FILE_ATTRIBUTE_OFFLINE: i32 = 4096;
pub const FILE_ATTRIBUTE_PINNED: i32 = 524288;
pub const FILE_ATTRIBUTE_READONLY: i32 = 1;
pub const FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS: i32 = 4194304;
pub const FILE_ATTRIBUTE_RECALL_ON_OPEN: i32 = 262144;
pub const FILE_ATTRIBUTE_REPARSE_POINT: i32 = 1024;
pub const FILE_ATTRIBUTE_SPARSE_FILE: i32 = 512;
pub const FILE_ATTRIBUTE_SYSTEM: i32 = 4;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_ATTRIBUTE_TAG_INFO {
    pub FileAttributes: u32,
    pub ReparseTag: u32,
}
pub const FILE_ATTRIBUTE_TEMPORARY: i32 = 256;
pub const FILE_ATTRIBUTE_UNPINNED: i32 = 1048576;
pub const FILE_ATTRIBUTE_VIRTUAL: i32 = 65536;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_BASIC_INFO {
    pub CreationTime: i64,
    pub LastAccessTime: i64,
    pub LastWriteTime: i64,
    pub ChangeTime: i64,
    pub FileAttributes: u32,
}
pub const FILE_BEGIN: i32 = 0;
pub const FILE_COMPLETE_IF_OPLOCKED: i32 = 256;
pub const FILE_CONTAINS_EXTENDED_CREATE_INFORMATION: i32 = 268435456;
pub const FILE_CREATE: i32 = 2;
pub const FILE_CREATE_PIPE_INSTANCE: i32 = 4;
pub const FILE_CREATE_TREE_CONNECTION: i32 = 128;
pub const FILE_CURRENT: i32 = 1;
pub const FILE_DELETE_CHILD: i32 = 64;
pub const FILE_DELETE_ON_CLOSE: i32 = 4096;
pub const FILE_DIRECTORY_FILE: i32 = 1;
pub const FILE_DISALLOW_EXCLUSIVE: i32 = 131072;
pub const FILE_DISPOSITION_FLAG_DELETE: i32 = 1;
pub const FILE_DISPOSITION_FLAG_DO_NOT_DELETE: i32 = 0;
pub const FILE_DISPOSITION_FLAG_FORCE_IMAGE_SECTION_CHECK: i32 = 4;
pub const FILE_DISPOSITION_FLAG_IGNORE_READONLY_ATTRIBUTE: i32 = 16;
pub const FILE_DISPOSITION_FLAG_ON_CLOSE: i32 = 8;
pub const FILE_DISPOSITION_FLAG_POSIX_SEMANTICS: i32 = 2;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_DISPOSITION_INFO {
    pub DeleteFile: bool,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_DISPOSITION_INFO_EX {
    pub Flags: u32,
}
pub const FILE_END: i32 = 2;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_END_OF_FILE_INFO {
    pub EndOfFile: i64,
}
pub const FILE_EXECUTE: i32 = 32;
pub const FILE_FLAG_BACKUP_SEMANTICS: i32 = 33554432;
pub const FILE_FLAG_DELETE_ON_CLOSE: i32 = 67108864;
pub const FILE_FLAG_FIRST_PIPE_INSTANCE: i32 = 524288;
pub const FILE_FLAG_NO_BUFFERING: i32 = 536870912;
pub const FILE_FLAG_OPEN_NO_RECALL: i32 = 1048576;
pub const FILE_FLAG_OPEN_REPARSE_POINT: i32 = 2097152;
pub const FILE_FLAG_OVERLAPPED: i32 = 1073741824;
pub const FILE_FLAG_POSIX_SEMANTICS: i32 = 16777216;
pub const FILE_FLAG_RANDOM_ACCESS: i32 = 268435456;
pub const FILE_FLAG_SEQUENTIAL_SCAN: i32 = 134217728;
pub const FILE_FLAG_SESSION_AWARE: i32 = 8388608;
pub const FILE_FLAG_WRITE_THROUGH: u32 = 2147483648;
pub const FILE_GENERIC_EXECUTE: i32 = 1179808;
pub const FILE_GENERIC_READ: i32 = 1179785;
pub const FILE_GENERIC_WRITE: i32 = 1179926;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FILE_ID_BOTH_DIR_INFO {
    pub NextEntryOffset: u32,
    pub FileIndex: u32,
    pub CreationTime: i64,
    pub LastAccessTime: i64,
    pub LastWriteTime: i64,
    pub ChangeTime: i64,
    pub EndOfFile: i64,
    pub AllocationSize: i64,
    pub FileAttributes: u32,
    pub FileNameLength: u32,
    pub EaSize: u32,
    pub ShortNameLength: CCHAR,
    pub ShortName: [u16; 12],
    pub FileId: i64,
    pub FileName: [u16; 1],
}
impl Default for FILE_ID_BOTH_DIR_INFO {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub type FILE_INFORMATION_CLASS = i32;
pub type FILE_INFO_BY_HANDLE_CLASS = i32;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_IO_PRIORITY_HINT_INFO {
    pub PriorityHint: PRIORITY_HINT,
}
pub const FILE_LIST_DIRECTORY: i32 = 1;
pub const FILE_NAME_NORMALIZED: i32 = 0;
pub const FILE_NAME_OPENED: i32 = 8;
pub const FILE_NON_DIRECTORY_FILE: i32 = 64;
pub const FILE_NO_COMPRESSION: i32 = 32768;
pub const FILE_NO_EA_KNOWLEDGE: i32 = 512;
pub const FILE_NO_INTERMEDIATE_BUFFERING: i32 = 8;
pub const FILE_OPEN: i32 = 1;
pub const FILE_OPEN_BY_FILE_ID: i32 = 8192;
pub const FILE_OPEN_FOR_BACKUP_INTENT: i32 = 16384;
pub const FILE_OPEN_FOR_FREE_SPACE_QUERY: i32 = 8388608;
pub const FILE_OPEN_IF: i32 = 3;
pub const FILE_OPEN_NO_RECALL: i32 = 4194304;
pub const FILE_OPEN_REPARSE_POINT: i32 = 2097152;
pub const FILE_OPEN_REQUIRING_OPLOCK: i32 = 65536;
pub const FILE_OVERWRITE: i32 = 4;
pub const FILE_OVERWRITE_IF: i32 = 5;
pub const FILE_PIPE_ACCEPT_REMOTE_CLIENTS: i32 = 0;
pub const FILE_PIPE_BYTE_STREAM_MODE: i32 = 0;
pub const FILE_PIPE_BYTE_STREAM_TYPE: i32 = 0;
pub const FILE_PIPE_COMPLETE_OPERATION: i32 = 1;
pub const FILE_PIPE_MESSAGE_MODE: i32 = 1;
pub const FILE_PIPE_MESSAGE_TYPE: i32 = 1;
pub const FILE_PIPE_QUEUE_OPERATION: i32 = 0;
pub const FILE_PIPE_REJECT_REMOTE_CLIENTS: i32 = 2;
pub const FILE_RANDOM_ACCESS: i32 = 2048;
pub const FILE_READ_ATTRIBUTES: i32 = 128;
pub const FILE_READ_DATA: i32 = 1;
pub const FILE_READ_EA: i32 = 8;
pub const FILE_RENAME_FLAG_POSIX_SEMANTICS: i32 = 2;
pub const FILE_RENAME_FLAG_REPLACE_IF_EXISTS: i32 = 1;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FILE_RENAME_INFO {
    pub Anonymous: FILE_RENAME_INFO_0,
    pub RootDirectory: HANDLE,
    pub FileNameLength: u32,
    pub FileName: [u16; 1],
}
impl Default for FILE_RENAME_INFO {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union FILE_RENAME_INFO_0 {
    pub ReplaceIfExists: bool,
    pub Flags: u32,
}
impl Default for FILE_RENAME_INFO_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FILE_RENAME_INFORMATION {
    pub Anonymous: FILE_RENAME_INFORMATION_0,
    pub RootDirectory: HANDLE,
    pub FileNameLength: u32,
    pub FileName: [u16; 1],
}
impl Default for FILE_RENAME_INFORMATION {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union FILE_RENAME_INFORMATION_0 {
    pub ReplaceIfExists: bool,
    pub Flags: u32,
}
impl Default for FILE_RENAME_INFORMATION_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const FILE_RESERVE_OPFILTER: i32 = 1048576;
pub const FILE_SEQUENTIAL_ONLY: i32 = 4;
pub const FILE_SESSION_AWARE: i32 = 262144;
pub const FILE_SHARE_DELETE: i32 = 4;
pub const FILE_SHARE_READ: i32 = 1;
pub const FILE_SHARE_WRITE: i32 = 2;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct FILE_STANDARD_INFO {
    pub AllocationSize: i64,
    pub EndOfFile: i64,
    pub NumberOfLinks: u32,
    pub DeletePending: bool,
    pub Directory: bool,
}
pub const FILE_SUPERSEDE: i32 = 0;
pub const FILE_SYNCHRONOUS_IO_ALERT: i32 = 16;
pub const FILE_SYNCHRONOUS_IO_NONALERT: i32 = 32;
pub const FILE_TRAVERSE: i32 = 32;
pub const FILE_TYPE_CHAR: i32 = 2;
pub const FILE_TYPE_DISK: i32 = 1;
pub const FILE_TYPE_PIPE: i32 = 3;
pub const FILE_TYPE_REMOTE: i32 = 32768;
pub const FILE_TYPE_UNKNOWN: i32 = 0;
pub const FILE_WRITE_ATTRIBUTES: i32 = 256;
pub const FILE_WRITE_DATA: i32 = 2;
pub const FILE_WRITE_EA: i32 = 16;
pub const FILE_WRITE_THROUGH: i32 = 2;
pub type FINDEX_INFO_LEVELS = i32;
pub type FINDEX_SEARCH_OPS = i32;
pub const FIONBIO: u32 = 2147772030;
#[repr(C)]
#[cfg(target_arch = "x86")]
#[derive(Clone, Copy)]
pub struct FLOATING_SAVE_AREA {
    pub ControlWord: u32,
    pub StatusWord: u32,
    pub TagWord: u32,
    pub ErrorOffset: u32,
    pub ErrorSelector: u32,
    pub DataOffset: u32,
    pub DataSelector: u32,
    pub RegisterArea: [u8; 80],
    pub Spare0: u32,
}
#[cfg(target_arch = "x86")]
impl Default for FLOATING_SAVE_AREA {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const FLS_OUT_OF_INDEXES: u32 = 4294967295;
pub const FORMAT_MESSAGE_ALLOCATE_BUFFER: i32 = 256;
pub const FORMAT_MESSAGE_ARGUMENT_ARRAY: i32 = 8192;
pub const FORMAT_MESSAGE_FROM_HMODULE: i32 = 2048;
pub const FORMAT_MESSAGE_FROM_STRING: i32 = 1024;
pub const FORMAT_MESSAGE_FROM_SYSTEM: i32 = 4096;
pub const FORMAT_MESSAGE_IGNORE_INSERTS: i32 = 512;
pub const FRS_ERR_SYSVOL_POPULATE_TIMEOUT: i32 = 8014;
pub const FSCTL_GET_REPARSE_POINT: i32 = 589992;
pub const FSCTL_SET_REPARSE_POINT: i32 = 589988;
pub const FileAlignmentInfo: FILE_INFO_BY_HANDLE_CLASS = 17;
pub const FileAllocationInfo: FILE_INFO_BY_HANDLE_CLASS = 5;
pub const FileAttributeTagInfo: FILE_INFO_BY_HANDLE_CLASS = 9;
pub const FileBasicInfo: FILE_INFO_BY_HANDLE_CLASS = 0;
pub const FileCaseSensitiveInfo: FILE_INFO_BY_HANDLE_CLASS = 23;
pub const FileCompressionInfo: FILE_INFO_BY_HANDLE_CLASS = 8;
pub const FileDispositionInfo: FILE_INFO_BY_HANDLE_CLASS = 4;
pub const FileDispositionInfoEx: FILE_INFO_BY_HANDLE_CLASS = 21;
pub const FileEndOfFileInfo: FILE_INFO_BY_HANDLE_CLASS = 6;
pub const FileFullDirectoryInfo: FILE_INFO_BY_HANDLE_CLASS = 14;
pub const FileFullDirectoryRestartInfo: FILE_INFO_BY_HANDLE_CLASS = 15;
pub const FileIdBothDirectoryInfo: FILE_INFO_BY_HANDLE_CLASS = 10;
pub const FileIdBothDirectoryRestartInfo: FILE_INFO_BY_HANDLE_CLASS = 11;
pub const FileIdExtdDirectoryInfo: FILE_INFO_BY_HANDLE_CLASS = 19;
pub const FileIdExtdDirectoryRestartInfo: FILE_INFO_BY_HANDLE_CLASS = 20;
pub const FileIdInfo: FILE_INFO_BY_HANDLE_CLASS = 18;
pub const FileIoPriorityHintInfo: FILE_INFO_BY_HANDLE_CLASS = 12;
pub const FileNameInfo: FILE_INFO_BY_HANDLE_CLASS = 2;
pub const FileNormalizedNameInfo: FILE_INFO_BY_HANDLE_CLASS = 24;
pub const FileRemoteProtocolInfo: FILE_INFO_BY_HANDLE_CLASS = 13;
pub const FileRenameInfo: FILE_INFO_BY_HANDLE_CLASS = 3;
pub const FileRenameInfoEx: FILE_INFO_BY_HANDLE_CLASS = 22;
pub const FileRenameInformation: FILE_INFORMATION_CLASS = 10;
pub const FileRenameInformationEx: FILE_INFORMATION_CLASS = 65;
pub const FileStandardInfo: FILE_INFO_BY_HANDLE_CLASS = 1;
pub const FileStorageInfo: FILE_INFO_BY_HANDLE_CLASS = 16;
pub const FileStreamInfo: FILE_INFO_BY_HANDLE_CLASS = 7;
pub const FindExInfoBasic: FINDEX_INFO_LEVELS = 1;
pub const FindExSearchNameMatch: FINDEX_SEARCH_OPS = 0;
pub const GENERIC_ALL: i32 = 268435456;
pub const GENERIC_EXECUTE: i32 = 536870912;
pub const GENERIC_READ: u32 = 2147483648;
pub const GENERIC_WRITE: i32 = 1073741824;
pub const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: i32 = 4;
pub const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: i32 = 2;
pub type GROUP = u32;
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GUID {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}
pub type HANDLE = *mut core::ffi::c_void;
pub const HANDLE_FLAG_INHERIT: i32 = 1;
pub const HANDLE_FLAG_PROTECT_FROM_CLOSE: i32 = 2;
pub const HIGH_PRIORITY_CLASS: i32 = 128;
pub type HINSTANCE = *mut core::ffi::c_void;
pub type HLOCAL = HANDLE;
pub type HMODULE = HINSTANCE;
pub type HRESULT = i32;
pub const IDLE_PRIORITY_CLASS: i32 = 64;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IN6_ADDR {
    pub u: IN6_ADDR_0,
}
impl Default for IN6_ADDR {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union IN6_ADDR_0 {
    pub Byte: [u8; 16],
    pub Word: [u16; 8],
}
impl Default for IN6_ADDR_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const INFINITE: u32 = 4294967295;
pub const INHERIT_CALLER_PRIORITY: i32 = 131072;
pub const INHERIT_PARENT_AFFINITY: i32 = 65536;
pub type INIT_ONCE = RTL_RUN_ONCE;
pub const INIT_ONCE_INIT_FAILED: u32 = 4;
pub const INVALID_FILE_ATTRIBUTES: u32 = 4294967295;
pub const INVALID_SOCKET: SOCKET = 18446744073709551615u64 as usize;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IN_ADDR {
    pub S_un: IN_ADDR_0,
}
impl Default for IN_ADDR {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union IN_ADDR_0 {
    pub S_un_b: IN_ADDR_0_0,
    pub S_un_w: IN_ADDR_0_1,
    pub S_addr: u32,
}
impl Default for IN_ADDR_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IN_ADDR_0_0 {
    pub s_b1: u8,
    pub s_b2: u8,
    pub s_b3: u8,
    pub s_b4: u8,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct IN_ADDR_0_1 {
    pub s_w1: u16,
    pub s_w2: u16,
}
pub const IO_REPARSE_TAG_MOUNT_POINT: u32 = 2684354563;
pub const IO_REPARSE_TAG_SYMLINK: u32 = 2684354572;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IO_STATUS_BLOCK {
    pub Anonymous: IO_STATUS_BLOCK_0,
    pub Information: usize,
}
impl Default for IO_STATUS_BLOCK {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union IO_STATUS_BLOCK_0 {
    pub Status: NTSTATUS,
    pub Pointer: *mut core::ffi::c_void,
}
impl Default for IO_STATUS_BLOCK_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub type IPPROTO = i32;
pub const IPPROTO_AH: IPPROTO = 51;
pub const IPPROTO_CBT: IPPROTO = 7;
pub const IPPROTO_DSTOPTS: IPPROTO = 60;
pub const IPPROTO_EGP: IPPROTO = 8;
pub const IPPROTO_ESP: IPPROTO = 50;
pub const IPPROTO_FRAGMENT: IPPROTO = 44;
pub const IPPROTO_GGP: IPPROTO = 3;
pub const IPPROTO_HOPOPTS: IPPROTO = 0;
pub const IPPROTO_ICLFXBM: IPPROTO = 78;
pub const IPPROTO_ICMP: IPPROTO = 1;
pub const IPPROTO_ICMPV6: IPPROTO = 58;
pub const IPPROTO_IDP: IPPROTO = 22;
pub const IPPROTO_IGMP: IPPROTO = 2;
pub const IPPROTO_IGP: IPPROTO = 9;
pub const IPPROTO_IP: i32 = 0;
pub const IPPROTO_IPV4: IPPROTO = 4;
pub const IPPROTO_IPV6: IPPROTO = 41;
pub const IPPROTO_L2TP: IPPROTO = 115;
pub const IPPROTO_MAX: IPPROTO = 256;
pub const IPPROTO_ND: IPPROTO = 77;
pub const IPPROTO_NONE: IPPROTO = 59;
pub const IPPROTO_PGM: IPPROTO = 113;
pub const IPPROTO_PIM: IPPROTO = 103;
pub const IPPROTO_PUP: IPPROTO = 12;
pub const IPPROTO_RAW: IPPROTO = 255;
pub const IPPROTO_RDP: IPPROTO = 27;
pub const IPPROTO_RESERVED_IPSEC: IPPROTO = 258;
pub const IPPROTO_RESERVED_IPSECOFFLOAD: IPPROTO = 259;
pub const IPPROTO_RESERVED_MAX: IPPROTO = 261;
pub const IPPROTO_RESERVED_RAW: IPPROTO = 257;
pub const IPPROTO_RESERVED_WNV: IPPROTO = 260;
pub const IPPROTO_RM: i32 = 113;
pub const IPPROTO_ROUTING: IPPROTO = 43;
pub const IPPROTO_SCTP: IPPROTO = 132;
pub const IPPROTO_ST: IPPROTO = 5;
pub const IPPROTO_TCP: IPPROTO = 6;
pub const IPPROTO_UDP: IPPROTO = 17;
pub const IPV6_ADD_MEMBERSHIP: i32 = 12;
pub const IPV6_DROP_MEMBERSHIP: i32 = 13;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IPV6_MREQ {
    pub ipv6mr_multiaddr: IN6_ADDR,
    pub ipv6mr_interface: u32,
}
impl Default for IPV6_MREQ {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const IPV6_MULTICAST_LOOP: i32 = 11;
pub const IPV6_V6ONLY: i32 = 27;
pub const IP_ADD_MEMBERSHIP: i32 = 12;
pub const IP_DROP_MEMBERSHIP: i32 = 13;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IP_MREQ {
    pub imr_multiaddr: IN_ADDR,
    pub imr_interface: IN_ADDR,
}
impl Default for IP_MREQ {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const IP_MULTICAST_LOOP: i32 = 11;
pub const IP_MULTICAST_TTL: i32 = 10;
pub const IP_TTL: i32 = 4;
pub type LINGER = linger;
pub const LOCKFILE_EXCLUSIVE_LOCK: i32 = 2;
pub const LOCKFILE_FAIL_IMMEDIATELY: i32 = 1;
pub type LPBYTE = *mut u8;
pub type LPINIT_ONCE = PRTL_RUN_ONCE;
pub type LPOVERLAPPED_COMPLETION_ROUTINE = Option<
    unsafe extern "system" fn(
        dwerrorcode: u32,
        dwnumberofbytestransfered: u32,
        lpoverlapped: *mut OVERLAPPED,
    ),
>;
pub type LPPROC_THREAD_ATTRIBUTE_LIST = *mut _PROC_THREAD_ATTRIBUTE_LIST;
pub type LPPROGRESS_ROUTINE = Option<
    unsafe extern "system" fn(
        totalfilesize: i64,
        totalbytestransferred: i64,
        streamsize: i64,
        streambytestransferred: i64,
        dwstreamnumber: u32,
        dwcallbackreason: u32,
        hsourcefile: HANDLE,
        hdestinationfile: HANDLE,
        lpdata: *const core::ffi::c_void,
    ) -> u32,
>;
pub type LPTHREAD_START_ROUTINE = PTHREAD_START_ROUTINE;
pub type LPWCH = *mut u16;
pub type LPWSAOVERLAPPED_COMPLETION_ROUTINE = Option<
    unsafe extern "system" fn(
        dwerror: u32,
        cbtransferred: u32,
        lpoverlapped: *mut OVERLAPPED,
        dwflags: u32,
    ),
>;
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct M128A {
    pub Low: u64,
    pub High: i64,
}
pub const MAXIMUM_REPARSE_DATA_BUFFER_SIZE: i32 = 16384;
pub const MAX_PATH: i32 = 260;
pub const MB_COMPOSITE: i32 = 2;
pub const MB_ERR_INVALID_CHARS: i32 = 8;
pub const MB_PRECOMPOSED: i32 = 1;
pub const MB_USEGLYPHCHARS: i32 = 4;
pub const MOVEFILE_COPY_ALLOWED: i32 = 2;
pub const MOVEFILE_CREATE_HARDLINK: i32 = 16;
pub const MOVEFILE_DELAY_UNTIL_REBOOT: i32 = 4;
pub const MOVEFILE_FAIL_IF_NOT_TRACKABLE: i32 = 32;
pub const MOVEFILE_REPLACE_EXISTING: i32 = 1;
pub const MOVEFILE_WRITE_THROUGH: i32 = 8;
pub const MSG_DONTROUTE: i32 = 4;
pub const MSG_OOB: i32 = 1;
pub const MSG_PEEK: i32 = 2;
pub const MSG_PUSH_IMMEDIATE: i32 = 32;
pub const MSG_WAITALL: i32 = 8;
pub const MaximumFileInfoByHandleClass: FILE_INFO_BY_HANDLE_CLASS = 25;
#[cfg(target_arch = "aarch64")]
pub type NEON128 = ARM64_NT_NEON128;
pub const NORMAL_PRIORITY_CLASS: i32 = 32;
pub const NO_ERROR: i32 = 0;
pub type NTSTATUS = i32;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OBJECT_ATTRIBUTES {
    pub Length: u32,
    pub RootDirectory: HANDLE,
    pub ObjectName: PUNICODE_STRING,
    pub Attributes: u32,
    pub SecurityDescriptor: *mut core::ffi::c_void,
    pub SecurityQualityOfService: *mut core::ffi::c_void,
}
impl Default for OBJECT_ATTRIBUTES {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const OBJ_CASE_INSENSITIVE: i32 = 64;
pub const OBJ_DONT_REPARSE: i32 = 4096;
pub const OBJ_EXCLUSIVE: i32 = 32;
pub const OBJ_FORCE_ACCESS_CHECK: i32 = 1024;
pub const OBJ_IGNORE_IMPERSONATED_DEVICEMAP: i32 = 2048;
pub const OBJ_INHERIT: i32 = 2;
pub const OBJ_KERNEL_HANDLE: i32 = 512;
pub const OBJ_OPENIF: i32 = 128;
pub const OBJ_OPENLINK: i32 = 256;
pub const OBJ_PERMANENT: i32 = 16;
pub const OPEN_ALWAYS: i32 = 4;
pub const OPEN_EXISTING: i32 = 3;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OVERLAPPED {
    pub Internal: usize,
    pub InternalHigh: usize,
    pub Anonymous: OVERLAPPED_0,
    pub hEvent: HANDLE,
}
impl Default for OVERLAPPED {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union OVERLAPPED_0 {
    pub Anonymous: OVERLAPPED_0_0,
    pub Pointer: *mut core::ffi::c_void,
}
impl Default for OVERLAPPED_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct OVERLAPPED_0_0 {
    pub Offset: u32,
    pub OffsetHigh: u32,
}
pub type PADDRINFOA = *mut ADDRINFOA;
#[cfg(any(target_arch = "arm64ec", target_arch = "x86", target_arch = "x86_64"))]
pub type PCONTEXT = *mut CONTEXT;
#[cfg(target_arch = "aarch64")]
pub type PCONTEXT = *mut ARM64_NT_CONTEXT;
pub type PCSTR = *const u8;
pub type PCWSTR = *const u16;
pub type PEXCEPTION_RECORD = *mut EXCEPTION_RECORD;
pub type PFLS_CALLBACK_FUNCTION =
    Option<unsafe extern "system" fn(lpflsdata: *const core::ffi::c_void)>;
pub type PIO_APC_ROUTINE = Option<
    unsafe extern "system" fn(
        apccontext: *mut core::ffi::c_void,
        iostatusblock: *mut IO_STATUS_BLOCK,
        reserved: u32,
    ),
>;
pub const PIPE_ACCEPT_REMOTE_CLIENTS: i32 = 0;
pub const PIPE_ACCESS_DUPLEX: i32 = 3;
pub const PIPE_ACCESS_INBOUND: i32 = 1;
pub const PIPE_ACCESS_OUTBOUND: i32 = 2;
pub const PIPE_CLIENT_END: i32 = 0;
pub const PIPE_NOWAIT: i32 = 1;
pub const PIPE_READMODE_BYTE: i32 = 0;
pub const PIPE_READMODE_MESSAGE: i32 = 2;
pub const PIPE_REJECT_REMOTE_CLIENTS: i32 = 8;
pub const PIPE_SERVER_END: i32 = 1;
pub const PIPE_TYPE_BYTE: i32 = 0;
pub const PIPE_TYPE_MESSAGE: i32 = 4;
pub const PIPE_WAIT: i32 = 0;
pub type PRIORITY_HINT = i32;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PROCESS_INFORMATION {
    pub hProcess: HANDLE,
    pub hThread: HANDLE,
    pub dwProcessId: u32,
    pub dwThreadId: u32,
}
impl Default for PROCESS_INFORMATION {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const PROCESS_MODE_BACKGROUND_BEGIN: i32 = 1048576;
pub const PROCESS_MODE_BACKGROUND_END: i32 = 2097152;
pub const PROFILE_KERNEL: i32 = 536870912;
pub const PROFILE_SERVER: i32 = 1073741824;
pub const PROFILE_USER: i32 = 268435456;
pub const PROGRESS_CONTINUE: i32 = 0;
pub type PRTL_RUN_ONCE = *mut RTL_RUN_ONCE;
pub type PSTR = *mut u8;
pub type PTHREAD_START_ROUTINE =
    Option<unsafe extern "system" fn(lpthreadparameter: *mut core::ffi::c_void) -> u32>;
pub type PTIMERAPCROUTINE = Option<
    unsafe extern "system" fn(
        lpargtocompletionroutine: *const core::ffi::c_void,
        dwtimerlowvalue: u32,
        dwtimerhighvalue: u32,
    ),
>;
pub type PUNICODE_STRING = *mut UNICODE_STRING;
pub type PVECTORED_EXCEPTION_HANDLER =
    Option<unsafe extern "system" fn(exceptioninfo: *mut EXCEPTION_POINTERS) -> i32>;
pub type PWSTR = *mut u16;
pub const READ_CONTROL: i32 = 131072;
pub const REALTIME_PRIORITY_CLASS: i32 = 256;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RTL_CONDITION_VARIABLE {
    pub Ptr: *mut core::ffi::c_void,
}
impl Default for RTL_CONDITION_VARIABLE {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union RTL_RUN_ONCE {
    pub Ptr: *mut core::ffi::c_void,
}
impl Default for RTL_RUN_ONCE {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RTL_SRWLOCK {
    pub Ptr: *mut core::ffi::c_void,
}
impl Default for RTL_SRWLOCK {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const SD_BOTH: i32 = 2;
pub const SD_RECEIVE: i32 = 0;
pub const SD_SEND: i32 = 1;
pub const SECURITY_ANONYMOUS: i32 = 0;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SECURITY_ATTRIBUTES {
    pub nLength: u32,
    pub lpSecurityDescriptor: *mut core::ffi::c_void,
    pub bInheritHandle: BOOL,
}
impl Default for SECURITY_ATTRIBUTES {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const SECURITY_CONTEXT_TRACKING: i32 = 262144;
pub const SECURITY_DELEGATION: i32 = 196608;
pub const SECURITY_EFFECTIVE_ONLY: i32 = 524288;
pub const SECURITY_IDENTIFICATION: i32 = 65536;
pub const SECURITY_IMPERSONATION: i32 = 131072;
pub const SECURITY_SQOS_PRESENT: i32 = 1048576;
pub const SECURITY_VALID_SQOS_FLAGS: i32 = 2031616;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SOCKADDR {
    pub sa_family: ADDRESS_FAMILY,
    pub sa_data: [i8; 14],
}
impl Default for SOCKADDR {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub type SOCKADDR_STORAGE = SOCKADDR_STORAGE_LH;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SOCKADDR_STORAGE_LH {
    pub ss_family: ADDRESS_FAMILY,
    pub __ss_pad1: [i8; 6],
    pub __ss_align: i64,
    pub __ss_pad2: [i8; 112],
}
impl Default for SOCKADDR_STORAGE_LH {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SOCKADDR_UN {
    pub sun_family: ADDRESS_FAMILY,
    pub sun_path: [i8; 108],
}
impl Default for SOCKADDR_UN {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub type SOCKET = usize;
pub const SOCKET_ERROR: i32 = -1;
pub const SOCK_DGRAM: i32 = 2;
pub const SOCK_RAW: i32 = 3;
pub const SOCK_RDM: i32 = 4;
pub const SOCK_SEQPACKET: i32 = 5;
pub const SOCK_STREAM: i32 = 1;
pub const SOL_SOCKET: i32 = 65535;
pub const SO_BROADCAST: i32 = 32;
pub const SO_ERROR: i32 = 4103;
pub const SO_KEEPALIVE: i32 = 8;
pub const SO_LINGER: i32 = 128;
pub const SO_RCVTIMEO: i32 = 4102;
pub const SO_SNDTIMEO: i32 = 4101;
pub const SPECIFIC_RIGHTS_ALL: i32 = 65535;
pub type SRWLOCK = RTL_SRWLOCK;
pub const STACK_SIZE_PARAM_IS_A_RESERVATION: i32 = 65536;
pub const STANDARD_RIGHTS_ALL: i32 = 2031616;
pub const STANDARD_RIGHTS_EXECUTE: i32 = 131072;
pub const STANDARD_RIGHTS_READ: i32 = 131072;
pub const STANDARD_RIGHTS_REQUIRED: i32 = 983040;
pub const STANDARD_RIGHTS_WRITE: i32 = 131072;
pub const STARTF_FORCEOFFFEEDBACK: i32 = 128;
pub const STARTF_FORCEONFEEDBACK: i32 = 64;
pub const STARTF_PREVENTPINNING: i32 = 8192;
pub const STARTF_RUNFULLSCREEN: i32 = 32;
pub const STARTF_TITLEISAPPID: i32 = 4096;
pub const STARTF_TITLEISLINKNAME: i32 = 2048;
pub const STARTF_UNTRUSTEDSOURCE: i32 = 32768;
pub const STARTF_USECOUNTCHARS: i32 = 8;
pub const STARTF_USEFILLATTRIBUTE: i32 = 16;
pub const STARTF_USEHOTKEY: i32 = 512;
pub const STARTF_USEPOSITION: i32 = 4;
pub const STARTF_USESHOWWINDOW: i32 = 1;
pub const STARTF_USESIZE: i32 = 2;
pub const STARTF_USESTDHANDLES: i32 = 256;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct STARTUPINFOEXW {
    pub StartupInfo: STARTUPINFOW,
    pub lpAttributeList: LPPROC_THREAD_ATTRIBUTE_LIST,
}
impl Default for STARTUPINFOEXW {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct STARTUPINFOW {
    pub cb: u32,
    pub lpReserved: PWSTR,
    pub lpDesktop: PWSTR,
    pub lpTitle: PWSTR,
    pub dwX: u32,
    pub dwY: u32,
    pub dwXSize: u32,
    pub dwYSize: u32,
    pub dwXCountChars: u32,
    pub dwYCountChars: u32,
    pub dwFillAttribute: u32,
    pub dwFlags: u32,
    pub wShowWindow: u16,
    pub cbReserved2: u16,
    pub lpReserved2: LPBYTE,
    pub hStdInput: HANDLE,
    pub hStdOutput: HANDLE,
    pub hStdError: HANDLE,
}
impl Default for STARTUPINFOW {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const STATUS_DELETE_PENDING: NTSTATUS = 0xC0000056_u32 as _;
pub const STATUS_DIRECTORY_NOT_EMPTY: NTSTATUS = 0xC0000101_u32 as _;
pub const STATUS_END_OF_FILE: NTSTATUS = 0xC0000011_u32 as _;
pub const STATUS_FILE_DELETED: NTSTATUS = 0xC0000123_u32 as _;
pub const STATUS_INVALID_HANDLE: NTSTATUS = 0xC0000008_u32 as _;
pub const STATUS_INVALID_PARAMETER: NTSTATUS = 0xC000000D_u32 as _;
pub const STATUS_NOT_IMPLEMENTED: NTSTATUS = 0xC0000002_u32 as _;
pub const STATUS_PENDING: NTSTATUS = 0x103_u32 as _;
pub const STATUS_SHARING_VIOLATION: NTSTATUS = 0xC0000043_u32 as _;
pub const STATUS_SUCCESS: NTSTATUS = 0x0_u32 as _;
pub const STD_ERROR_HANDLE: u32 = 4294967284;
pub const STD_INPUT_HANDLE: u32 = 4294967286;
pub const STD_OUTPUT_HANDLE: u32 = 4294967285;
pub const SYMBOLIC_LINK_FLAG_ALLOW_UNPRIVILEGED_CREATE: i32 = 2;
pub const SYMBOLIC_LINK_FLAG_DIRECTORY: i32 = 1;
pub const SYMLINK_FLAG_RELATIVE: i32 = 1;
pub const SYNCHRONIZE: i32 = 1048576;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SYSTEM_INFO {
    pub Anonymous: SYSTEM_INFO_0,
    pub dwPageSize: u32,
    pub lpMinimumApplicationAddress: *mut core::ffi::c_void,
    pub lpMaximumApplicationAddress: *mut core::ffi::c_void,
    pub dwActiveProcessorMask: usize,
    pub dwNumberOfProcessors: u32,
    pub dwProcessorType: u32,
    pub dwAllocationGranularity: u32,
    pub wProcessorLevel: u16,
    pub wProcessorRevision: u16,
}
impl Default for SYSTEM_INFO {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub union SYSTEM_INFO_0 {
    pub dwOemId: u32,
    pub Anonymous: SYSTEM_INFO_0_0,
}
impl Default for SYSTEM_INFO_0 {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SYSTEM_INFO_0_0 {
    pub wProcessorArchitecture: u16,
    pub wReserved: u16,
}
pub const TCP_NODELAY: i32 = 1;
pub const TIMER_ALL_ACCESS: i32 = 2031619;
pub const TIMER_MODIFY_STATE: i32 = 2;
pub type TIMEVAL = timeval;
pub const TLS_OUT_OF_INDEXES: u32 = 4294967295;
pub const TOKEN_ACCESS_PSEUDO_HANDLE: i32 = 24;
pub const TOKEN_ACCESS_PSEUDO_HANDLE_WIN8: i32 = 24;
pub const TOKEN_ADJUST_DEFAULT: i32 = 128;
pub const TOKEN_ADJUST_GROUPS: i32 = 64;
pub const TOKEN_ADJUST_PRIVILEGES: i32 = 32;
pub const TOKEN_ADJUST_SESSIONID: i32 = 256;
pub const TOKEN_ALL_ACCESS: i32 = 983551;
pub const TOKEN_ASSIGN_PRIMARY: i32 = 1;
pub const TOKEN_DUPLICATE: i32 = 2;
pub const TOKEN_EXECUTE: i32 = 131072;
pub const TOKEN_IMPERSONATE: i32 = 4;
pub const TOKEN_QUERY: i32 = 8;
pub const TOKEN_QUERY_SOURCE: i32 = 16;
pub const TOKEN_READ: i32 = 131080;
pub const TOKEN_TRUST_CONSTRAINT_MASK: i32 = 131096;
pub const TOKEN_WRITE: i32 = 131296;
pub const TRUE: i32 = 1;
pub const TRUNCATE_EXISTING: i32 = 5;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct UNICODE_STRING {
    pub Length: u16,
    pub MaximumLength: u16,
    pub Buffer: PWSTR,
}
impl Default for UNICODE_STRING {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const VOLUME_NAME_DOS: i32 = 0;
pub const VOLUME_NAME_GUID: i32 = 1;
pub const VOLUME_NAME_NONE: i32 = 4;
pub const WAIT_ABANDONED: i32 = 128;
pub const WAIT_ABANDONED_0: i32 = 128;
pub const WAIT_FAILED: u32 = 4294967295;
pub const WAIT_IO_COMPLETION: i32 = 192;
pub const WAIT_OBJECT_0: i32 = 0;
pub const WAIT_TIMEOUT: i32 = 258;
pub const WC_ERR_INVALID_CHARS: i32 = 128;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WIN32_FIND_DATAW {
    pub dwFileAttributes: u32,
    pub ftCreationTime: FILETIME,
    pub ftLastAccessTime: FILETIME,
    pub ftLastWriteTime: FILETIME,
    pub nFileSizeHigh: u32,
    pub nFileSizeLow: u32,
    pub dwReserved0: u32,
    pub dwReserved1: u32,
    pub cFileName: [u16; 260],
    pub cAlternateFileName: [u16; 14],
}
impl Default for WIN32_FIND_DATAW {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const WRITE_DAC: i32 = 262144;
pub const WRITE_OWNER: i32 = 524288;
pub const WSABASEERR: i32 = 10000;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WSABUF {
    pub len: u32,
    pub buf: *mut i8,
}
impl Default for WSABUF {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(target_arch = "x86")]
#[derive(Clone, Copy)]
pub struct WSADATA {
    pub wVersion: u16,
    pub wHighVersion: u16,
    pub szDescription: [i8; 257],
    pub szSystemStatus: [i8; 129],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut i8,
}
#[cfg(target_arch = "x86")]
impl Default for WSADATA {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct WSADATA {
    pub wVersion: u16,
    pub wHighVersion: u16,
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: *mut i8,
    pub szDescription: [i8; 257],
    pub szSystemStatus: [i8; 129],
}
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", target_arch = "x86_64"))]
impl Default for WSADATA {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const WSAEACCES: i32 = 10013;
pub const WSAEADDRINUSE: i32 = 10048;
pub const WSAEADDRNOTAVAIL: i32 = 10049;
pub const WSAEAFNOSUPPORT: i32 = 10047;
pub const WSAEALREADY: i32 = 10037;
pub const WSAEBADF: i32 = 10009;
pub const WSAECANCELLED: i32 = 10103;
pub const WSAECONNABORTED: i32 = 10053;
pub const WSAECONNREFUSED: i32 = 10061;
pub const WSAECONNRESET: i32 = 10054;
pub const WSAEDESTADDRREQ: i32 = 10039;
pub const WSAEDISCON: i32 = 10101;
pub const WSAEDQUOT: i32 = 10069;
pub const WSAEFAULT: i32 = 10014;
pub const WSAEHOSTDOWN: i32 = 10064;
pub const WSAEHOSTUNREACH: i32 = 10065;
pub const WSAEINPROGRESS: i32 = 10036;
pub const WSAEINTR: i32 = 10004;
pub const WSAEINVAL: i32 = 10022;
pub const WSAEINVALIDPROCTABLE: i32 = 10104;
pub const WSAEINVALIDPROVIDER: i32 = 10105;
pub const WSAEISCONN: i32 = 10056;
pub const WSAELOOP: i32 = 10062;
pub const WSAEMFILE: i32 = 10024;
pub const WSAEMSGSIZE: i32 = 10040;
pub const WSAENAMETOOLONG: i32 = 10063;
pub const WSAENETDOWN: i32 = 10050;
pub const WSAENETRESET: i32 = 10052;
pub const WSAENETUNREACH: i32 = 10051;
pub const WSAENOBUFS: i32 = 10055;
pub const WSAENOMORE: i32 = 10102;
pub const WSAENOPROTOOPT: i32 = 10042;
pub const WSAENOTCONN: i32 = 10057;
pub const WSAENOTEMPTY: i32 = 10066;
pub const WSAENOTSOCK: i32 = 10038;
pub const WSAEOPNOTSUPP: i32 = 10045;
pub const WSAEPFNOSUPPORT: i32 = 10046;
pub const WSAEPROCLIM: i32 = 10067;
pub const WSAEPROTONOSUPPORT: i32 = 10043;
pub const WSAEPROTOTYPE: i32 = 10041;
pub const WSAEPROVIDERFAILEDINIT: i32 = 10106;
pub const WSAEREFUSED: i32 = 10112;
pub const WSAEREMOTE: i32 = 10071;
pub const WSAESHUTDOWN: i32 = 10058;
pub const WSAESOCKTNOSUPPORT: i32 = 10044;
pub const WSAESTALE: i32 = 10070;
pub const WSAETIMEDOUT: i32 = 10060;
pub const WSAETOOMANYREFS: i32 = 10059;
pub const WSAEUSERS: i32 = 10068;
pub const WSAEWOULDBLOCK: i32 = 10035;
pub const WSAHOST_NOT_FOUND: i32 = 11001;
pub const WSANOTINITIALISED: i32 = 10093;
pub const WSANO_DATA: i32 = 11004;
pub const WSANO_RECOVERY: i32 = 11003;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WSAPROTOCOLCHAIN {
    pub ChainLen: i32,
    pub ChainEntries: [u32; 7],
}
impl Default for WSAPROTOCOLCHAIN {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WSAPROTOCOL_INFOW {
    pub dwServiceFlags1: u32,
    pub dwServiceFlags2: u32,
    pub dwServiceFlags3: u32,
    pub dwServiceFlags4: u32,
    pub dwProviderFlags: u32,
    pub ProviderId: GUID,
    pub dwCatalogEntryId: u32,
    pub ProtocolChain: WSAPROTOCOLCHAIN,
    pub iVersion: i32,
    pub iAddressFamily: i32,
    pub iMaxSockAddr: i32,
    pub iMinSockAddr: i32,
    pub iSocketType: i32,
    pub iProtocol: i32,
    pub iProtocolMaxOffset: i32,
    pub iNetworkByteOrder: i32,
    pub iSecurityScheme: i32,
    pub dwMessageSize: u32,
    pub dwProviderReserved: u32,
    pub szProtocol: [u16; 256],
}
impl Default for WSAPROTOCOL_INFOW {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
pub const WSASERVICE_NOT_FOUND: i32 = 10108;
pub const WSASYSCALLFAILURE: i32 = 10107;
pub const WSASYSNOTREADY: i32 = 10091;
pub const WSATRY_AGAIN: i32 = 11002;
pub const WSATYPE_NOT_FOUND: i32 = 10109;
pub const WSAVERNOTSUPPORTED: i32 = 10092;
pub const WSA_E_CANCELLED: i32 = 10111;
pub const WSA_E_NO_MORE: i32 = 10110;
pub const WSA_FLAG_NO_HANDLE_INHERIT: i32 = 128;
pub const WSA_FLAG_OVERLAPPED: i32 = 1;
pub const WSA_INVALID_HANDLE: i32 = 6;
pub const WSA_INVALID_PARAMETER: i32 = 87;
pub const WSA_IO_INCOMPLETE: i32 = 996;
pub const WSA_IO_PENDING: i32 = 997;
pub const WSA_IPSEC_NAME_POLICY_ERROR: i32 = 11033;
pub const WSA_NOT_ENOUGH_MEMORY: i32 = 8;
pub const WSA_OPERATION_ABORTED: i32 = 995;
pub const WSA_QOS_ADMISSION_FAILURE: i32 = 11010;
pub const WSA_QOS_BAD_OBJECT: i32 = 11013;
pub const WSA_QOS_BAD_STYLE: i32 = 11012;
pub const WSA_QOS_EFILTERCOUNT: i32 = 11021;
pub const WSA_QOS_EFILTERSTYLE: i32 = 11019;
pub const WSA_QOS_EFILTERTYPE: i32 = 11020;
pub const WSA_QOS_EFLOWCOUNT: i32 = 11023;
pub const WSA_QOS_EFLOWDESC: i32 = 11026;
pub const WSA_QOS_EFLOWSPEC: i32 = 11017;
pub const WSA_QOS_EOBJLENGTH: i32 = 11022;
pub const WSA_QOS_EPOLICYOBJ: i32 = 11025;
pub const WSA_QOS_EPROVSPECBUF: i32 = 11018;
pub const WSA_QOS_EPSFILTERSPEC: i32 = 11028;
pub const WSA_QOS_EPSFLOWSPEC: i32 = 11027;
pub const WSA_QOS_ESDMODEOBJ: i32 = 11029;
pub const WSA_QOS_ESERVICETYPE: i32 = 11016;
pub const WSA_QOS_ESHAPERATEOBJ: i32 = 11030;
pub const WSA_QOS_EUNKOWNPSOBJ: i32 = 11024;
pub const WSA_QOS_GENERIC_ERROR: i32 = 11015;
pub const WSA_QOS_NO_RECEIVERS: i32 = 11008;
pub const WSA_QOS_NO_SENDERS: i32 = 11007;
pub const WSA_QOS_POLICY_FAILURE: i32 = 11011;
pub const WSA_QOS_RECEIVERS: i32 = 11005;
pub const WSA_QOS_REQUEST_CONFIRMED: i32 = 11009;
pub const WSA_QOS_RESERVED_PETYPE: i32 = 11031;
pub const WSA_QOS_SENDERS: i32 = 11006;
pub const WSA_QOS_TRAFFIC_CTRL_ERROR: i32 = 11014;
pub const WSA_SECURE_HOST_NOT_FOUND: i32 = 11032;
pub const WSA_WAIT_EVENT_0: i32 = 0;
pub const WSA_WAIT_IO_COMPLETION: i32 = 192;
#[cfg(any(target_arch = "arm64ec", target_arch = "x86_64"))]
pub type XMM_SAVE_AREA32 = XSAVE_FORMAT;
#[repr(C)]
#[cfg(target_arch = "x86")]
#[derive(Clone, Copy)]
pub struct XSAVE_FORMAT {
    pub ControlWord: u16,
    pub StatusWord: u16,
    pub TagWord: u8,
    pub Reserved1: u8,
    pub ErrorOpcode: u16,
    pub ErrorOffset: u32,
    pub ErrorSelector: u16,
    pub Reserved2: u16,
    pub DataOffset: u32,
    pub DataSelector: u16,
    pub Reserved3: u16,
    pub MxCsr: u32,
    pub MxCsr_Mask: u32,
    pub FloatRegisters: [M128A; 8],
    pub XmmRegisters: [M128A; 8],
    pub Reserved4: [u8; 224],
}
#[cfg(target_arch = "x86")]
impl Default for XSAVE_FORMAT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", target_arch = "x86_64"))]
#[derive(Clone, Copy)]
pub struct XSAVE_FORMAT {
    pub ControlWord: u16,
    pub StatusWord: u16,
    pub TagWord: u8,
    pub Reserved1: u8,
    pub ErrorOpcode: u16,
    pub ErrorOffset: u32,
    pub ErrorSelector: u16,
    pub Reserved2: u16,
    pub DataOffset: u32,
    pub DataSelector: u16,
    pub Reserved3: u16,
    pub MxCsr: u32,
    pub MxCsr_Mask: u32,
    pub FloatRegisters: [M128A; 8],
    pub XmmRegisters: [M128A; 16],
    pub Reserved4: [u8; 96],
}
#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", target_arch = "x86_64"))]
impl Default for XSAVE_FORMAT {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct _PROC_THREAD_ATTRIBUTE_LIST(pub u8);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct fd_set {
    pub fd_count: u_int,
    pub fd_array: [SOCKET; 64],
}
impl Default for fd_set {
    fn default() -> Self {
        unsafe { core::mem::zeroed() }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct linger {
    pub l_onoff: u_short,
    pub l_linger: u_short,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct timeval {
    pub tv_sec: i32,
    pub tv_usec: i32,
}
pub type u_int = u32;
pub type u_long = u32;
pub type u_short = u16;
pub type va_list = *mut i8;

#[cfg(target_arch = "arm")]
#[repr(C)]
pub struct WSADATA {
    pub wVersion: u16,
    pub wHighVersion: u16,
    pub szDescription: [u8; 257],
    pub szSystemStatus: [u8; 129],
    pub iMaxSockets: u16,
    pub iMaxUdpDg: u16,
    pub lpVendorInfo: PSTR,
}
#[cfg(target_arch = "arm")]
pub enum CONTEXT {}
#[cfg(target_arch = "arm")]
pub type PCONTEXT = *mut CONTEXT;
// ignore-tidy-file-filelength
