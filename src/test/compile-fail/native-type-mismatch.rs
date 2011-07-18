// error-pattern:expected native but found native
use std;

fn main() {
    let std::os::libc::FILE f = std::io::rustrt::rust_get_stdin();
    std::os::libc::fopen(f, f);
}
