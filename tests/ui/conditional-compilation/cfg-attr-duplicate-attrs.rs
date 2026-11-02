// Test for handling of duplicated attributes within `cfg_attr`, in particular the suggestions of
// what to remove.

#[inline]
#[inline]
//~^ WARN unused attribute
//~| WARN this was previously accepted
fn f1() {}

#[deprecated]
#[deprecated]
//~^ ERROR multiple `deprecated` attributes
fn f2() {}

#[inline]
#[cfg_attr(true, inline)]
//~^ WARN unused attribute
//~| WARN this was previously accepted
fn f3() {}

#[deprecated]
#[cfg_attr(true, deprecated)]
//~^ ERROR multiple `deprecated` attributes
fn f4() {}

#[inline]
#[deprecated]
#[cfg_attr(true, inline, deprecated)]
//~^ WARN unused attribute
//~| WARN this was previously accepted
//~| ERROR multiple `deprecated` attributes
fn f5() {}

fn main() {}
