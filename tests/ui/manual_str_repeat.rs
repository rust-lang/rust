// run-rustfix

#![warn(clippy::manual_str_repeat)]

use std::iter::repeat;

fn main() {
    let _: String = std::iter::repeat("test").take(10).collect();
    let _: String = std::iter::repeat('x').take(10).collect();
    let _: String = std::iter::repeat('\'').take(10).collect();
    let _: String = std::iter::repeat('"').take(10).collect();

    let x = "test";
    let count = 10;
    let _ = repeat(x).take(count + 2).collect::<String>();

    macro_rules! m {
        ($e:expr) => {{ $e }};
    }

    let _: String = repeat(m!("test")).take(m!(count)).collect();

    let x = &x;
    let _: String = repeat(*x).take(count).collect();

    macro_rules! repeat_m {
        ($e:expr) => {{ repeat($e) }};
    }
    // Don't lint, repeat is from a macro.
    let _: String = repeat_m!("test").take(count).collect();
}
