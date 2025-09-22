// Regression test for #133271.
#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete

struct Foo;
impl<'a, const NUM: usize> std::ops::Add<&'a Foo> for Foo
//~^ ERROR the const parameter `NUM` is not constrained by the impl trait, self type, or predicates
//~| ERROR missing: `Output`, `add`
where
    [(); 1 + 0]: Sized,
{
    fn unimplemented(self, _: &Foo) -> Self::Output {
        //~^ ERROR method `unimplemented` is not a member of trait `std::ops::Add`
        loop {}
    }
}

fn main() {}
