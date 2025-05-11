//@ only-x86_64

trait Foo {
    fn foo(&self);
    unsafe fn unsf_foo(&self);
}

struct Bar;

impl Foo for Bar {
    #[target_feature(enable = "sse2")]
    //~^ ERROR cannot be applied to safe trait method
    fn foo(&self) {}
    //~^ ERROR method `foo` has an incompatible type for trait

    #[target_feature(enable = "sse2")]
    unsafe fn unsf_foo(&self) {}
}

trait Qux {
    #[target_feature(enable = "sse2")]
    //~^ ERROR cannot be applied to safe trait method
    fn foo(&self) {}
}

fn main() {}
