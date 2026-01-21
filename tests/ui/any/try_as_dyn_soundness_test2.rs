//@ run-pass
#![feature(try_as_dyn)]
use std::any::try_as_dyn;

trait Trait<T> {

}

impl Trait<for<'a> fn(&'a Box<i32>)> for () {

}

fn main() {
    let dt = try_as_dyn::<_, dyn Trait<fn(&'static Box<i32>)>>(&());
    assert!(dt.is_none());
}
