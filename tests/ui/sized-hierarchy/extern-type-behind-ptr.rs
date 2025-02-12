//@ check-pass
#![feature(extern_types, sized_hierarchy)]

pub fn hash<T: PointeeSized>(_: *const T) {
    unimplemented!();
}

unsafe extern "C" {
    type Foo;
}

fn get() -> *const Foo {
    unimplemented!()
}

fn main() {
    hash::<Foo>(get());
}
