//@ edition: 2021

// Reject raw lifetimes with identifier parts that wouldn't be valid raw identifiers.

macro_rules! w {
    ($tt:tt) => {};
}

w!('r#_);
//~^ ERROR `_` cannot be a raw lifetime
w!('r#self);
//~^ ERROR `self` cannot be a raw lifetime
w!('r#super);
//~^ ERROR `super` cannot be a raw lifetime
w!('r#Self);
//~^ ERROR `Self` cannot be a raw lifetime
w!('r#crate);
//~^ ERROR `crate` cannot be a raw lifetime

fn main() {}
