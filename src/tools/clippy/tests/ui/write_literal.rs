#![warn(clippy::write_literal)]
#![allow(clippy::uninlined_format_args, unused_must_use)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();

    // these should be fine
    write!(v, "Hello");
    writeln!(v, "Hello");
    let world = "world";
    writeln!(v, "Hello {}", world);
    writeln!(v, "Hello {world}", world = world);
    writeln!(v, "3 in hex is {:X}", 3);
    writeln!(v, "2 + 1 = {:.4}", 3);
    writeln!(v, "2 + 1 = {:5.4}", 3);
    writeln!(v, "Debug test {:?}", "hello, world");
    writeln!(v, "{0:8} {1:>8}", "hello", "world");
    writeln!(v, "{1:8} {0:>8}", "hello", "world");
    writeln!(v, "{foo:8} {bar:>8}", foo = "hello", bar = "world");
    writeln!(v, "{bar:8} {foo:>8}", foo = "hello", bar = "world");
    writeln!(v, "{number:>width$}", number = 1, width = 6);
    writeln!(v, "{number:>0width$}", number = 1, width = 6);
    writeln!(v, "{} of {:b} people know binary, the other half doesn't", 1, 2);
    writeln!(v, "10 / 4 is {}", 2.5);
    writeln!(v, "2 + 1 = {}", 3);
    writeln!(v, "From expansion {}", stringify!(not a string literal));

    // these should throw warnings
    write!(v, "Hello {}", "world");
    //~^ ERROR: literal with an empty format string
    //~| NOTE: `-D clippy::write-literal` implied by `-D warnings`
    writeln!(v, "Hello {} {}", world, "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "Hello {}", "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{} {:.4}", "a literal", 5);
    //~^ ERROR: literal with an empty format string

    // positional args don't change the fact
    // that we're using a literal -- this should
    // throw a warning
    writeln!(v, "{0} {1}", "hello", "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{1} {0}", "hello", "world");
    //~^ ERROR: literal with an empty format string

    // named args shouldn't change anything either
    writeln!(v, "{foo} {bar}", foo = "hello", bar = "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{bar} {foo}", foo = "hello", bar = "world");
    //~^ ERROR: literal with an empty format string

    // #10128
    writeln!(v, "{0} {1} {2}", "hello", 2, "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{2} {1} {0}", "hello", 2, "world");
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{0} {1} {2}, {bar}", "hello", 2, 3, bar = 4);
    //~^ ERROR: literal with an empty format string
    writeln!(v, "{0} {1} {2}, {3} {4}", "hello", 2, 3, "world", 4);
    //~^ ERROR: literal with an empty format string
}
