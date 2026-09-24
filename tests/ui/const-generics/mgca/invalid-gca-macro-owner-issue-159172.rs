//@ revisions: ty expr
#![feature(min_generic_const_args)]

use std::gca;

#[cfg(ty)]
trait Iter<
    const C: gca!(|| {
        //[ty]~^ ERROR expected type, found `gca!()` constant
        use std::io::*;
        let mut buffer = std::fs::File::create("foo.txt")?;
        write!(buffer, "oh no")?;
    }),
>
{
}

#[cfg(ty)]
fn main() {}

#[cfg(expr)]
fn main() {
    let _ = gca!(|| {
        //[expr]~^ ERROR expected expression, found `gca!()` constant
        use std::io::*;
        write!(_, "")
    });
}
