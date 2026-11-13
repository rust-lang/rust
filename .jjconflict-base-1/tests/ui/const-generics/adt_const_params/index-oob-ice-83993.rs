// issue: rust-lang/rust/#83993

#![feature(adt_const_params)]

fn bug<'a>()
where
    for<'b> [(); {
        let x: &'b ();
        //~^ ERROR generic parameters may not be used in const operations
        0
    }]:,
{
}

fn bad()
where
    for<'b> [(); {
        let _: &'b ();
        //~^ ERROR generic parameters may not be used in const operations
        0
    }]: Sized,
{
}
fn good()
where
    for<'b> [(); { 0 }]: Sized,
{
}

pub fn main() {}
