// only-x86_64

#![feature(target_feature_11)]

trait Foo {
    fn foo(&self);
    unsafe fn unsf_foo(&self);
}

struct Bar;

impl Foo for Bar {
    #[target_feature(enable = "sse2")]
    //~^ ERROR cannot be applied to safe trait method
    fn foo(&self) {}

    #[target_feature(enable = "sse2")]
    unsafe fn unsf_foo(&self) {}
}

fn main() {}
