// run-pass
//! Regression test for #34426, regarding HRTB in drop impls

pub trait Lifetime<'a> {}
impl<'a> Lifetime<'a> for i32 {}

#[allow(dead_code)]
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

pub trait Lifetime2<'a, 'b> {}
impl<'a, 'b> Lifetime2<'a, 'b> for i32 {}

#[allow(dead_code)]
struct Bar<L>
    where for<'a, 'b> L: Lifetime2<'a, 'b>
{
    l: L
}

impl<L> Drop for Bar<L>
    where for<'a, 'b> L: Lifetime2<'a, 'b>
{
    fn drop(&mut self) {
        println!("drop with hrtb");
    }
}

fn main() {
    let _foo = Foo {
        l: 0
    };

    let _bar = Bar {
        l: 0
    };
}
