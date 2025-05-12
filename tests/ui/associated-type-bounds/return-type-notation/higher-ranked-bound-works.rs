//@ check-pass

#![feature(return_type_notation)]

trait Trait<'a> {
    fn late<'b>(&'b self, _: &'a ()) -> impl Sized;
    fn early<'b: 'b>(&'b self, _: &'a ()) -> impl Sized;
}

#[allow(refining_impl_trait_internal)]
impl<'a> Trait<'a> for () {
    fn late<'b>(&'b self, _: &'a ()) -> i32 { 1 }
    fn early<'b: 'b>(&'b self, _: &'a ()) -> i32 { 1 }
}

trait Other<'c> {}
impl Other<'_> for i32 {}

fn test<T>(t: &T)
where
    T: for<'a, 'c> Trait<'a, late(..): Other<'c>>,
    // which is basically:
    // for<'a, 'c> Trait<'a, for<'b> method<'b>: Other<'c>>,
    T: for<'a, 'c> Trait<'a, early(..): Other<'c>>,
    // which is basically:
    // for<'a, 'c> Trait<'a, for<'b> method<'b>: Other<'c>>,
{
    is_other_impl(t.late(&()));
    is_other_impl(t.early(&()));
}

fn test_path<T>(t: &T)
where
T: for<'a> Trait<'a>,
    for<'a, 'c> <T as Trait<'a>>::late(..): Other<'c>,
    // which is basically:
    // for<'a, 'b, 'c> <T as Trait<'a>>::method::<'b>: Other<'c>
    for<'a, 'c> <T as Trait<'a>>::early(..): Other<'c>,
    // which is basically:
    // for<'a, 'b, 'c> <T as Trait<'a>>::method::<'b>: Other<'c>
{
    is_other_impl(t.late(&()));
    is_other_impl(t.early(&()));
}

fn is_other_impl(_: impl for<'c> Other<'c>) {}

fn main() {
    test(&());
    test(&());
}
