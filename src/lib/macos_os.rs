import _str.sbuf;
import _vec.vbuf;

native mod libc = "libc.dylib" {

  fn open(sbuf s, int flags) -> int;
  fn read(int fd, vbuf buf, uint count) -> int;
  fn write(int fd, vbuf buf, uint count) -> int;
  fn close(int fd) -> int;

  type dir;
  // readdir is a mess; handle via wrapper function in rustrt.
  fn opendir(sbuf d) -> dir;
  fn closedir(dir d) -> int;

  fn getenv(sbuf n) -> sbuf;
  fn setenv(sbuf n, sbuf v, int overwrite) -> int;
  fn unsetenv(sbuf n) -> int;
}
