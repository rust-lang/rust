#![feature(min_generic_const_args)]

use std::gca;

trait Iter<
    const FN: fn() = {
        gca!(|| {
            //~^ ERROR complex const arguments must be placed inside of a `const` block
            use std::io::*;
            write!(_, "")
        })
    },
>
{
}
