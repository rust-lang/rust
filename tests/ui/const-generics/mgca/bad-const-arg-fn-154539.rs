#![feature(min_generic_const_args)]

trait Iter<
    const FN: fn() = {
        || { //~ ERROR complex const arguments must be placed inside of a `const` block
            use std::io::*;
            write!(_, "")
        }
    },
>
{
}
