#![feature(rustc_attrs)]
#![feature(infer_static_outlives_requirements)]

#[rustc_outlives]
struct Foo<U> { //~ ERROR rustc_outlives
    bar: Bar<U>
}
struct Bar<T: 'static> {
    x: T,
}

fn main() {}
