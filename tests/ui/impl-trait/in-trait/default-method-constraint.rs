// check-pass

// This didn't work in the previous default RPITIT method hack attempt

#![feature(return_position_impl_trait_in_trait)]

trait Foo {
    fn bar(x: bool) -> impl Sized {
        if x {
            let _: u32 = Self::bar(!x);
        }
        Default::default()
    }
}

fn main() {}
