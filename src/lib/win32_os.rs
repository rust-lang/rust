import _str.sbuf;
import _vec.vbuf;

native mod libc = "msvcrt.dll" {
  fn open(sbuf s, int flags, uint mode) -> int = "_open";
  fn read(int fd, vbuf buf, uint count) -> int = "_read";
  fn write(int fd, vbuf buf, uint count) -> int = "_write";
  fn close(int fd) -> int = "_close";

  type FILE;
  fn fopen(sbuf path, sbuf mode) -> FILE;
  fn fclose(FILE f);
  fn fgetc(FILE f) -> int;
  fn ungetc(int c, FILE f);
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
