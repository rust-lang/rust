#![feature(intrinsics)]

trait Foo {
    extern "rust-intrinsic" fn foo(&self); //~ ERROR intrinsic must
}

impl Foo for () {
    extern "rust-intrinsic" fn foo(&self) { //~ ERROR intrinsic must
    }
}

extern "rust-intrinsic" fn hello() {//~ ERROR intrinsic must
}

fn main() {
}
