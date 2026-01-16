// Main crate that triggers panics in the dependency.
// Used to verify panic locations are correct with -Z separate-spans.

extern crate dep;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let test = args.get(1).map(|s| s.as_str()).unwrap_or("public");

    match test {
        "public" => dep::will_panic(true),
        "private" => dep::panic_via_private(true),
        "format" => dep::panic_with_format(-1),
        _ => panic!("unknown test: {}", test),
    }
}
