//@ run-pass
//@ aux-build:default_type_params_xc.rs


extern crate default_type_params_xc;

struct Vec<T, A = default_type_params_xc::Heap>(#[allow(dead_code)] Option<(T,A)>);

struct Foo;

fn main() {
    let _a = Vec::<isize>(None);
    let _b = Vec::<isize, default_type_params_xc::FakeHeap>(None);
    let _c = default_type_params_xc::FakeVec::<isize> { f: None };
    let _d = default_type_params_xc::FakeVec::<isize, Foo> { f: None };
}
