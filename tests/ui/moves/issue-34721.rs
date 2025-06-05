//@ run-rustfix

pub trait Foo {
    fn zero(self) -> Self;
}

impl Foo for u32 {
    fn zero(self) -> u32 { 0u32 }
}

pub mod bar {
    pub use crate::Foo;
    pub fn bar<T: Foo>(x: T) -> T {
      x.zero()
    }
}

mod baz {
    use crate::bar;
    use crate::Foo;
    pub fn baz<T: Foo>(x: T) -> T {
        if 0 == 1 {
            bar::bar(x.zero())
        } else {
            x.zero()
        };
        x.zero()
        //~^ ERROR use of moved value
    }
}

fn main() {
    let _ = baz::baz(0u32);
}
