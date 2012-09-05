// xfail-fast
// aux-build:issue-3012-1.rs
use socketlib;
use socketlib::socket;

fn main() {
    let fd: libc::c_int = 1 as libc::c_int;
    let sock = @socket::socket_handle(fd);
}
