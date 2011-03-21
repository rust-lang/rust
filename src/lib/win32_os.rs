import _str.sbuf;
import _vec.vbuf;

native mod libc = "msvcrt.dll" {
    fn open(sbuf s, int flags, uint mode) -> int = "_open";
    fn read(int fd, vbuf buf, uint count) -> int = "_read";
    fn write(int fd, vbuf buf, uint count) -> int = "_write";
    fn close(int fd) -> int = "_close";

    type FILE;
    fn fopen(sbuf path, sbuf mode) -> FILE;
    fn _fdopen(int fd, sbuf mode) -> FILE;
    fn fclose(FILE f);
    fn fgetc(FILE f) -> int;
    fn ungetc(int c, FILE f);
    fn fread(vbuf buf, uint size, uint n, FILE f) -> uint;
    fn fseek(FILE f, int offset, int whence) -> int;
    fn ftell(FILE f) -> int;

    fn _pipe(vbuf fds, uint size, int mode) -> int;
}

mod libc_constants {
    fn O_RDONLY() -> int { ret 0x0000; }
    fn O_WRONLY() -> int { ret 0x0001; }
    fn O_RDWR()   -> int { ret 0x0002; }
    fn O_APPEND() -> int { ret 0x0400; }
    fn O_CREAT()  -> int { ret 0x0040; }
    fn O_EXCL()   -> int { ret 0x0080; }
    fn O_TRUNC()  -> int { ret 0x0200; }
    fn O_TEXT()   -> int { ret 0x4000; }
    fn O_BINARY() -> int { ret 0x8000; }

    fn S_IRUSR() -> uint { ret 0x0100u; } // really _S_IREAD  in win32
    fn S_IWUSR() -> uint { ret 0x0080u; } // really _S_IWRITE in win32
}

fn exec_suffix() -> str {
    ret ".exe";
}

fn target_os() -> str {
    ret "win32";
}

fn dylib_filename(str base) -> str {
    ret base + ".dll";
}

fn pipe() -> tup(int, int) {
    let vec[mutable int] fds = vec(mutable 0, 0);
    check(os.libc._pipe(_vec.buf[mutable int](fds), 1024u,
                        libc_constants.O_BINARY()) == 0);
    ret tup(fds.(0), fds.(1));
}

fn fd_FILE(int fd) -> libc.FILE {
    ret libc._fdopen(fd, _str.buf("r"));
}

native "rust" mod rustrt {
    fn rust_process_wait(int handle) -> int;
}

fn waitpid(int pid) -> int {
    ret rustrt.rust_process_wait(pid);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
