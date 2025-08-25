//@ check-pass

#![feature(rustc_attrs)]
#![warn(unused)]

macro_rules! foo {
    () => {}
}

fn main() {
    #[rustc_dummy] foo!(); //~ WARN unused attribute `rustc_dummy`

    // This does nothing, since `#[allow(warnings)]` is itself
    // an inert attribute on a macro call
    #[allow(warnings)] #[rustc_dummy] foo!(); //~ WARN unused attribute `allow`
    //~^ WARN unused attribute `rustc_dummy`

    // This does work, since the attribute is on a parent
    // of the macro invocation.
    #[allow(warnings)] { #[rustc_dummy] foo!(); }

    // Ok, `cfg` and `cfg_attr` are expanded eagerly and do not warn.
    #[cfg(true)] foo!();
    #[cfg(false)] foo!();
    #[cfg_attr(true, cfg(true))] foo!();
    #[cfg_attr(false, nonexistent)] foo!();
}
