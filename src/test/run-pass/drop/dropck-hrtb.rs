// run-pass
//! Regression test for #34426, regarding HRTB in drop impls

pub trait Lifetime<'a> {}
impl<'a> Lifetime<'a> for i32 {}

struct Foo<L>
    where for<'a> L: Lifetime<'a>
{
    l: L
}

impl<L> Drop for Foo<L>
    where for<'a> L: Lifetime<'a>
{
    fn drop(&mut self) {
        println!("drop with hrtb");
    }
}

fn main() {
    let _foo = Foo {
        l: 0
    };
}
