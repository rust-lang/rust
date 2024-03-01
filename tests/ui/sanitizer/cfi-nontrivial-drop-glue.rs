//@ needs-sanitizer-cfi
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C codegen-units=1 -C opt-level=0
//@ run-pass

use std::env;

fn main() {
    env::current_exe().unwrap();
}
