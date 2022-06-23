// run-rustfix
#![allow(unused_must_use)]
#![allow(named_arguments_used_positionally)] // Unstable at time of writing.
#![warn(clippy::positional_named_format_parameters)]

use std::io::Write;

fn main() {
    let mut v = Vec::new();
    let hello = "Hello";

    println!("{hello:.foo$}", foo = 2);
    writeln!(v, "{hello:.foo$}", foo = 2);

    // Warnings
    println!("{} {1:?}", zero = 0, one = 1);
    println!("This is a test { } {000001:?}", zero = 0, one = 1);
    println!("Hello {1} is {2:.0$}", zero = 5, one = hello, two = 0.01);
    println!("Hello {1:0$}!", zero = 5, one = 1);
    println!("Hello {0:1$}!", zero = 4, one = 1);
    println!("Hello {0:01$}!", zero = 4, one = 1);
    println!("Hello is {1:.*}", zero = 5, one = 0.01);
    println!("Hello is {:<6.*}", zero = 5, one = 0.01);
    println!("{}, `{two:>8.*}` has 3", zero = hello, one = 3, two = hello);
    println!("Hello {1} is {2:.0$}", zero = 5, one = hello, two = 0.01);
    println!("Hello {world} {}!", world = 5);

    writeln!(v, "{} {1:?}", zero = 0, one = 1);
    writeln!(v, "This is a test { } {000001:?}", zero = 0, one = 1);
    writeln!(v, "Hello {1} is {2:.0$}", zero = 5, one = hello, two = 0.01);
    writeln!(v, "Hello {1:0$}!", zero = 4, one = 1);
    writeln!(v, "Hello {0:1$}!", zero = 4, one = 1);
    writeln!(v, "Hello {0:01$}!", zero = 4, one = 1);
    writeln!(v, "Hello is {1:.*}", zero = 3, one = 0.01);
    writeln!(v, "Hello is {:<6.*}", zero = 2, one = 0.01);
    writeln!(v, "{}, `{two:>8.*}` has 3", zero = hello, one = 3, two = hello);
    writeln!(v, "Hello {1} is {2:.0$}", zero = 1, one = hello, two = 0.01);
    writeln!(v, "Hello {world} {}!", world = 0);

    // Tests from other files
    println!("{:w$}", w = 1);
    println!("{:.p$}", p = 1);
    println!("{}", v = 1);
    println!("{:0$}", v = 1);
    println!("{0:0$}", v = 1);
    println!("{:0$.0$}", v = 1);
    println!("{0:0$.0$}", v = 1);
    println!("{0:0$.v$}", v = 1);
    println!("{0:v$.0$}", v = 1);
    println!("{v:0$.0$}", v = 1);
    println!("{v:v$.0$}", v = 1);
    println!("{v:0$.v$}", v = 1);
    println!("{:w$}", w = 1);
    println!("{:.p$}", p = 1);
    println!("{:p$.w$}", 1, w = 1, p = 1);
}
