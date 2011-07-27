


// -*- rust -*-
fn main() {
    let f = "Makefile";
    let s = rustrt.str_buf(f);
    let buf = libc.malloc(1024);
    let fd = libc.open(s, 0, 0);
    libc.read(fd, buf, 1024);
    libc.write(1, buf, 1024);
    libc.close(fd);
    libc.free(buf);
}