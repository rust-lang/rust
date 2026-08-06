//@ revisions: ty expr
#![feature(min_generic_const_args)]

#[cfg(ty)]
trait Iter<
    const C: core::direct_const_arg!(|| {
        //[ty]~^ ERROR expected type, found `direct_const_arg!()` constant
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
    let _ = core::direct_const_arg!(|| {
        //[expr]~^ ERROR expected expression, found `direct_const_arg!()` constant
        use std::io::*;
        write!(_, "")
    });
}
