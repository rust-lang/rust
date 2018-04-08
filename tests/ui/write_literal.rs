#![allow(unused_must_use)]
#![warn(write_literal)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // These should be fine
    write!(&mut v, "Hello");
    writeln!(&mut v, "Hello");
    let world = "world";
    writeln!(&mut v, "Hello {}", world);
    writeln!(&mut v, "3 in hex is {:X}", 3);

    // These should throw warnings
    write!(&mut v, "Hello {}", "world");
    writeln!(&mut v, "Hello {} {}", world, "world");
    writeln!(&mut v, "Hello {}", "world");
    writeln!(&mut v, "10 / 4 is {}", 2.5);
    writeln!(&mut v, "2 + 1 = {}", 3);
    writeln!(&mut v, "2 + 1 = {:.4}", 3);
    writeln!(&mut v, "2 + 1 = {:5.4}", 3);
    writeln!(&mut v, "Debug test {:?}", "hello, world");

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    writeln!(&mut v, "{0} {1}", "hello", "world");
    writeln!(&mut v, "{1} {0}", "hello", "world");

    // named args shouldn't change anything either
    writeln!(&mut v, "{foo} {bar}", foo = "hello", bar = "world");
    writeln!(&mut v, "{bar} {foo}", foo = "hello", bar = "world");
}
