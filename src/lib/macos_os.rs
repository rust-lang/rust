
import str::sbuf;
import vec::vbuf;

native "cdecl" mod libc = "c" {
    fn open(sbuf s, int flags, uint mode) -> int;
    fn read(int fd, vbuf buf, uint count) -> int;
    fn write(int fd, vbuf buf, uint count) -> int;
    fn close(int fd) -> int;
    type FILE;
    fn fopen(sbuf path, sbuf mode) -> FILE;
    fn fdopen(int fd, sbuf mode) -> FILE;
    fn fclose(FILE f);
    fn fgetc(FILE f) -> int;
    fn ungetc(int c, FILE f);
    fn feof(FILE f) -> int;
    fn fread(vbuf buf, uint size, uint n, FILE f) -> uint;
    fn fwrite(vbuf buf, uint size, uint n, FILE f) -> uint;
    fn fseek(FILE f, int offset, int whence) -> int;
    fn ftell(FILE f) -> int;
    type dir;
    fn opendir(sbuf d) -> dir;
    fn closedir(dir d) -> int;
    type dirent;
    fn readdir(dir d) -> dirent;
    fn getenv(sbuf n) -> sbuf;
    fn setenv(sbuf n, sbuf v, int overwrite) -> int;
    fn unsetenv(sbuf n) -> int;
    fn pipe(vbuf buf) -> int;
    fn waitpid(int pid, vbuf status, int options) -> int;
}

mod libc_constants {
    fn O_RDONLY() -> int { ret 0; }
    fn O_WRONLY() -> int { ret 1; }
    fn O_RDWR() -> int { ret 2; }
    fn O_APPEND() -> int { ret 8; }
    fn O_CREAT() -> int { ret 512; }
    fn O_EXCL() -> int { ret 2048; }
    fn O_TRUNC() -> int { ret 1024; }
    fn O_TEXT() -> int {
        ret 0; // nonexistent in darwin libc

    }
    fn O_BINARY() -> int {
        ret 0; // nonexistent in darwin libc

    }
    fn S_IRUSR() -> uint { ret 1024u; }
    fn S_IWUSR() -> uint { ret 512u; }
}

fn exec_suffix() -> str { ret ""; }

fn target_os() -> str { ret "macos"; }

fn dylib_filename(str base) -> str { ret "lib" + base + ".dylib"; }

fn pipe() -> tup(int, int) {
    let vec[mutable int] fds = [mutable 0, 0];
    assert (os::libc::pipe(vec::buf(fds)) == 0);
    ret tup(fds.(0), fds.(1));
}

fn fd_FILE(int fd) -> libc::FILE { ret libc::fdopen(fd, str::buf("r")); }

fn waitpid(int pid) -> int {
    let vec[mutable int] status = [mutable 0];
    assert (os::libc::waitpid(pid, vec::buf(status), 0) != -1);
    ret status.(0);
}

native "rust" mod rustrt {
    fn rust_getcwd() -> str;
}

fn getcwd() -> str { ret rustrt::rust_getcwd(); }


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
