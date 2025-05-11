#![feature(intrinsics)]

trait Foo {
    extern "rust-intrinsic" fn foo(&self); //~ ERROR invalid ABI
}

impl Foo for () {
    extern "rust-intrinsic" fn foo(&self) { //~ ERROR invalid ABI
    }
}

extern "rust-intrinsic" fn hello() { //~ ERROR invalid ABI
}

extern "rust-intrinsic" {
    //~^ ERROR invalid ABI
}

fn main() {}
