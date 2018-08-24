#![feature(rustc_attrs)]
#![feature(infer_outlives_requirements)]
#![feature(infer_static_outlives_requirements)]

#[rustc_outlives]
struct Foo<U> { //~ ERROR 16:1: 18:2: rustc_outlives
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}

fn main() {}

