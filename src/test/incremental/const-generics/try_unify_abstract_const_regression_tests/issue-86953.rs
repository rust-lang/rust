// revisions: rpass
#![feature(const_generics, const_evaluatable_checked)]
#![allow(incomplete_features)]

struct Foo;
impl<'a> std::ops::Add<&'a Foo> for Foo
where
    [(); 0 + 0]: Sized,
{
    type Output = ();
    fn add(self, _: &Foo) -> Self::Output {
        loop {}
    }
}

fn main() {}
