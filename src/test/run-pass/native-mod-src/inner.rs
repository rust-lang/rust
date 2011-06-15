


// -*- rust -*-
fn main() {
    auto f = "Makefile";
    auto s = rustrt.str_buf(f);
    auto buf = libc.malloc(1024);
    auto fd = libc.open(s, 0, 0);
    libc.read(fd, buf, 1024);
    libc.write(1, buf, 1024);
    libc.close(fd);
    libc.free(buf);
}