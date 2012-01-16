// error-pattern:expected `sbuf` but found `FILE`
use std;

fn main() unsafe {
    let f: std::os::libc::FILE = std::io::rustrt::rust_get_stdin();
    std::os::libc::fopen(f, f);
}
