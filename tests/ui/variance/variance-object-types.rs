#![feature(rustc_attrs)]


// For better or worse, associated types are invariant, and hence we
// get an invariant result for `'a`.
#[rustc_variance]
struct Foo<'a> { //~ ERROR ['a: o]
    x: Box<dyn Fn(i32) -> &'a i32 + 'static>
}

fn main() {
}
