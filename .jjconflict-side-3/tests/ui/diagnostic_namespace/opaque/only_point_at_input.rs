//@revisions: default -Zmacro-backtrace
//@[-Zmacro-backtrace] compile-flags: -Z macro-backtrace
//@ aux-crate:wrap=wrap.rs
#![feature(diagnostic_opaque)]

mod blah {}

#[diagnostic::opaque]
macro_rules! local_wrap {
    ($x:ident) => {{
        let x = blah::$x;
    }};
}


fn main() {
    wrap::wrap!(x);
    //~^ ERROR cannot find value `x` in module `blah`

    local_wrap!(x);
    //~^ ERROR cannot find value `x` in module `blah`

}
