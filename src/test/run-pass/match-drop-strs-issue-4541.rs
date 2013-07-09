// Tests a tricky scenario involving string matching,
// copying, and moving to ensure that we don't segfault
// or double-free, as we were wont to do in the past.

use std::io;
use std::os;

fn parse_args() -> ~str {
    let args = os::args();
    let mut n = 0;

    while n < args.len() {
        match copy args[n] {
            ~"-v" => (),
            s => {
                return s;
            }
        }
        n += 1;
    }

    return ~""
}

fn main() {
    io::println(parse_args());
}
