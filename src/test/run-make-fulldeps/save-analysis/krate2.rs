#![ crate_name = "krate2" ]
#![ crate_type = "lib" ]

use std::io::Write;

pub fn hello() {
    std::io::stdout().write_all(b"hello world!\n");
}
