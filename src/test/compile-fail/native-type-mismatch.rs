// error-pattern:expected native but found native
use std;

fn main() {
    let f: std::os::libc::FILE = std::io::rustrt::rust_get_stdin();
    std::os::libc::fopen(f, f);
}
