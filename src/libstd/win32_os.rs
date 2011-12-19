import core::option;
import core::ctypes::*;

#[abi = "cdecl"]
#[link_name = ""]               // FIXME remove after #[nolink] is snapshotted
#[nolink]
native mod libc {
    fn read(fd: fd_t, buf: *u8, count: size_t) -> ssize_t;
    fn write(fd: fd_t, buf: *u8, count: size_t) -> ssize_t;
    fn fread(buf: *u8, size: size_t, n: size_t, f: libc::FILE) -> size_t;
    fn fwrite(buf: *u8, size: size_t, n: size_t, f: libc::FILE) -> size_t;
    #[link_name = "_open"]
    fn open(s: str::sbuf, flags: c_int, mode: unsigned) -> c_int;
    #[link_name = "_close"]
    fn close(fd: fd_t) -> c_int;
    type FILE;
    fn fopen(path: str::sbuf, mode: str::sbuf) -> FILE;
    fn _fdopen(fd: fd_t, mode: str::sbuf) -> FILE;
    fn fclose(f: FILE);
    fn fflush(f: FILE) -> c_int;
    fn fileno(f: FILE) -> fd_t;
    fn fgetc(f: FILE) -> c_int;
    fn ungetc(c: c_int, f: FILE);
    fn feof(f: FILE) -> c_int;
    fn fseek(f: FILE, offset: long, whence: c_int) -> c_int;
    fn ftell(f: FILE) -> long;
    fn _pipe(fds: *mutable fd_t, size: unsigned, mode: c_int) -> c_int;
}

mod libc_constants {
    const O_RDONLY: c_int    = 0i32;
    const O_WRONLY: c_int    = 1i32;
    const O_RDWR: c_int      = 2i32;
    const O_APPEND: c_int    = 8i32;
    const O_CREAT: c_int     = 256i32;
    const O_EXCL: c_int      = 1024i32;
    const O_TRUNC: c_int     = 512i32;
    const O_TEXT: c_int      = 16384i32;
    const O_BINARY: c_int    = 32768i32;
    const O_NOINHERIT: c_int = 128i32;
    const S_IRUSR: unsigned  = 256u32; // really _S_IREAD  in win32
    const S_IWUSR: unsigned  = 128u32; // really _S_IWRITE in win32
}

type DWORD = u32;
type HMODULE = uint;
type LPTSTR = str::sbuf;
type LPCTSTR = str::sbuf;

#[abi = "stdcall"]
native mod kernel32 {
    type LPSECURITY_ATTRIBUTES;
    fn GetEnvironmentVariableA(n: str::sbuf, v: str::sbuf, nsize: uint) ->
       uint;
    fn SetEnvironmentVariableA(n: str::sbuf, v: str::sbuf) -> int;
    fn GetModuleFileNameA(hModule: HMODULE,
                          lpFilename: LPTSTR,
                          nSize: DWORD) -> DWORD;
    fn CreateDirectoryA(lpPathName: LPCTSTR,
                        lpSecurityAttributes: LPSECURITY_ATTRIBUTES) -> bool;
    fn RemoveDirectoryA(lpPathName: LPCTSTR) -> bool;
    fn SetCurrentDirectoryA(lpPathName: LPCTSTR) -> bool;
}

// FIXME turn into constants
fn exec_suffix() -> str { ret ".exe"; }
fn target_os() -> str { ret "win32"; }

fn dylib_filename(base: str) -> str { ret base + ".dll"; }

fn pipe() -> {in: fd_t, out: fd_t} {
    // Windows pipes work subtly differently than unix pipes, and their
    // inheritance has to be handled in a different way that I don't fully
    // understand. Here we explicitly make the pipe non-inheritable,
    // which means to pass it to a subprocess they need to be duplicated
    // first, as in rust_run_program.
    let fds = {mutable in: 0i32, mutable out: 0i32};
    let res =
        os::libc::_pipe(ptr::mut_addr_of(fds.in), 1024u32,
                        libc_constants::O_BINARY |
                            libc_constants::O_NOINHERIT);
    assert (res == 0i32);
    assert (fds.in != -1i32 && fds.in != 0i32);
    assert (fds.out != -1i32 && fds.in != 0i32);
    ret {in: fds.in, out: fds.out};
}

fn fd_FILE(fd: fd_t) -> libc::FILE {
    ret str::as_buf("r", {|modebuf| libc::_fdopen(fd, modebuf) });
}

fn close(fd: fd_t) -> c_int {
    libc::close(fd)
}

fn fclose(file: libc::FILE) {
    libc::fclose(file)
}

fn fsync_fd(fd: fd_t, level: io::fsync::level) -> c_int {
    // FIXME (1253)
    fail;
}

#[abi = "cdecl"]
native mod rustrt {
    fn rust_process_wait(handle: c_int) -> c_int;
    fn rust_getcwd() -> str;
}

fn waitpid(pid: pid_t) -> i32 { ret rustrt::rust_process_wait(pid); }

fn getcwd() -> str { ret rustrt::rust_getcwd(); }

fn get_exe_path() -> option::t<fs::path> {
    // FIXME: This doesn't handle the case where the buffer is too small
    let bufsize = 1023u;
    let path = str::unsafe_from_bytes(vec::init_elt(0u8, bufsize));
    ret str::as_buf(path, { |path_buf|
        if kernel32::GetModuleFileNameA(0u, path_buf,
                                        bufsize as u32) != 0u32 {
            option::some(fs::dirname(path) + fs::path_sep())
        } else {
            option::none
        }
    });
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
