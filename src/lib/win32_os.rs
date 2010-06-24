import _str.sbuf;
import _vec.vbuf;

native mod libc = "msvcrt.dll" {
  fn open(sbuf s, int flags) -> int = "_open";
  fn read(int fd, vbuf buf, uint count) -> int = "_read";
  fn write(int fd, vbuf buf, uint count) -> int = "_write";
  fn close(int fd) -> int = "_close";
}
