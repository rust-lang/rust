//@ check-pass

#![warn(unused)]

macro_rules! foo {
    () => {}
}

fn main() {
    #[inline] foo!(); //~ WARN `#[inline]` attribute cannot be used on macro calls
    //~^ WARN previously accepted

    // This does nothing, since `#[allow(warnings)]` is itself
    // an inert attribute on a macro call
    #[allow(warnings)] #[inline] foo!(); //~ WARN unused attribute `allow`
    //~^ WARN `#[inline]` attribute cannot be used on macro calls
    //~| WARN previously accepted

    // This does work, since the attribute is on a parent
    // of the macro invocation.
    #[allow(warnings)] { #[inline] foo!(); }

    // Ok, `cfg` and `cfg_attr` are expanded eagerly and do not warn.
    #[cfg(true)] foo!();
    #[cfg(false)] foo!();
    #[cfg_attr(true, cfg(true))] foo!();
    #[cfg_attr(false, nonexistent)] foo!();
}
