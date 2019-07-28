// error-pattern:thread 'main' panicked at 'foobar'

use std::panic;

fn main() {
    let _ = panic::take_hook();
    panic!("foobar");
}
