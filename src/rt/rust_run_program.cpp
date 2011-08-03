#include "rust_internal.h"

#if defined(__WIN32__)

#include <process.h>
#include <io.h>

extern "C" CDECL int
rust_run_program(void* task, const char* argv[],
                 int in_fd, int out_fd, int err_fd) {
    STARTUPINFO si;
    ZeroMemory(&si, sizeof(STARTUPINFO));
    si.cb = sizeof(STARTUPINFO);
    si.dwFlags = STARTF_USESTDHANDLES;

    HANDLE curproc = GetCurrentProcess();
    HANDLE origStdin = (HANDLE)_get_osfhandle(in_fd ? in_fd : 0);
    if (!DuplicateHandle(curproc, origStdin,
        curproc, &si.hStdInput, 0, 1, DUPLICATE_SAME_ACCESS))
        return -1;
    HANDLE origStdout = (HANDLE)_get_osfhandle(out_fd ? out_fd : 1);
    if (!DuplicateHandle(curproc, origStdout,
        curproc, &si.hStdOutput, 0, 1, DUPLICATE_SAME_ACCESS))
        return -1;
    HANDLE origStderr = (HANDLE)_get_osfhandle(err_fd ? err_fd : 2);
    if (!DuplicateHandle(curproc, origStderr,
        curproc, &si.hStdError, 0, 1, DUPLICATE_SAME_ACCESS))
        return -1;

    size_t cmd_len = 0;
    for (const char** arg = argv; *arg; arg++) {
        cmd_len += strlen(*arg);
        cmd_len++; // Space or \0
    }
    char* cmd = (char*)malloc(cmd_len);
    char* pos = cmd;
    for (const char** arg = argv; *arg; arg++) {
        strcpy(pos, *arg);
        pos += strlen(*arg);
        if (*(arg+1)) *(pos++) = ' ';
    }

    PROCESS_INFORMATION pi;
    BOOL created = CreateProcess(NULL, cmd, NULL, NULL, TRUE,
                                 0, NULL, NULL, &si, &pi);
                                 
    CloseHandle(si.hStdInput);
    CloseHandle(si.hStdOutput);
    CloseHandle(si.hStdError);
    free(cmd);

    if (!created) return -1;
    return (int)pi.hProcess;
}

extern "C" CDECL int
rust_process_wait(void* task, int proc) {
    DWORD status;
    while (true) {
        if (GetExitCodeProcess((HANDLE)proc, &status) &&
            status != STILL_ACTIVE)
            return (int)status;
        WaitForSingleObject((HANDLE)proc, INFINITE);
    }
}

#elif defined(__GNUC__)

#include <sys/file.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>

extern "C" CDECL int
rust_run_program(rust_task* task, char* argv[],
                 int in_fd, int out_fd, int err_fd) {
    int pid = fork();
    if (pid != 0) return pid;

    sigset_t sset;
    sigemptyset(&sset);
    sigprocmask(SIG_SETMASK, &sset, NULL);

    if (in_fd) dup2(in_fd, 0);
    if (out_fd) dup2(out_fd, 1);
    if (err_fd) dup2(err_fd, 2);
    /* Close all other fds. */
    for (int fd = getdtablesize() - 1; fd >= 3; fd--) close(fd);
    execvp(argv[0], argv);
    exit(1);
}

extern "C" CDECL int
rust_process_wait(void* task, int proc) {
    // FIXME: stub; exists to placate linker.
    return 0;
}

#else
#error "Platform not supported."
#endif

//
// Local Variables:
// mode: C++
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//
