// error-pattern:expected '*Mb' but found 'native'
use std;

fn main() unsafe {
    let f: std::os::libc::FILE = std::io::rustrt::rust_get_stdin();
    std::os::libc::fopen(f, f);
}
