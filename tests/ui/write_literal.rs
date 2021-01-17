#![allow(unused_must_use)]
#![warn(clippy::write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // these should be fine
    write!(&mut v, "Hello");
    writeln!(&mut v, "Hello");
    let world = "world";
    writeln!(&mut v, "Hello {}", world);
    writeln!(&mut v, "Hello {world}", world = world);
    writeln!(&mut v, "3 in hex is {:X}", 3);
    writeln!(&mut v, "2 + 1 = {:.4}", 3);
    writeln!(&mut v, "2 + 1 = {:5.4}", 3);
    writeln!(&mut v, "Debug test {:?}", "hello, world");
    writeln!(&mut v, "{0:8} {1:>8}", "hello", "world");
    writeln!(&mut v, "{1:8} {0:>8}", "hello", "world");
    writeln!(&mut v, "{foo:8} {bar:>8}", foo = "hello", bar = "world");
    writeln!(&mut v, "{bar:8} {foo:>8}", foo = "hello", bar = "world");
    writeln!(&mut v, "{number:>width$}", number = 1, width = 6);
    writeln!(&mut v, "{number:>0width$}", number = 1, width = 6);
    writeln!(&mut v, "{} of {:b} people know binary, the other half doesn't", 1, 2);
    writeln!(&mut v, "10 / 4 is {}", 2.5);
    writeln!(&mut v, "2 + 1 = {}", 3);

    // these should throw warnings
    write!(&mut v, "Hello {}", "world");
    writeln!(&mut v, "Hello {} {}", world, "world");
    writeln!(&mut v, "Hello {}", "world");

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    writeln!(&mut v, "{0} {1}", "hello", "world");
    writeln!(&mut v, "{1} {0}", "hello", "world");

    // named args shouldn't change anything either
    writeln!(&mut v, "{foo} {bar}", foo = "hello", bar = "world");
    writeln!(&mut v, "{bar} {foo}", foo = "hello", bar = "world");
}
