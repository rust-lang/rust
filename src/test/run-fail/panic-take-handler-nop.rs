// error-pattern:thread 'main' panicked at 'foobar'

use std::panic;

fn main() {
    panic::take_hook();
    panic!("foobar");
}
