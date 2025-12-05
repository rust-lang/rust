#![feature(fn_traits)]

fn id<T>(x: T) -> T { x }

pub fn foo<'a, F: Fn(&'a ())>(bar: F) {
    bar.call((
        &id(()), //~ ERROR temporary value dropped while borrowed
    ));
}
fn main() {}
