// FIXME(more_maybe_bounds): Even under `more_maybe_bounds` / `-Zexperimental-default-bounds`,
// trying to relax non-default bounds should still be an error in all contexts! As you can see
// there are places like supertrait bounds, trait object types or associated type bounds (ATB)
// where we currently don't perform this check.
#![feature(auto_traits, more_maybe_bounds, negative_impls)]

trait Trait1 {}
auto trait Trait2 {}

// FIXME: `?Trait1` should be rejected, `Trait1` isn't marked `#[lang = "default_traitN"]`.
trait Trait3: ?Trait1 {}
trait Trait4 where Self: Trait1 {}

// FIXME: `?Trait2` should be rejected, `Trait2` isn't marked `#[lang = "default_traitN"]`.
fn foo(_: Box<(dyn Trait3 + ?Trait2)>) {}

fn bar<T: ?Sized + ?Trait2 + ?Trait1 + ?Trait4>(_: &T) {}
//~^ ERROR bound modifier `?` can only be applied to default traits like `Sized`
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`

// FIXME: `?Trait1` should be rejected, `Trait1` isn't marked `#[lang = "default_traitN"]`.
fn baz<T>() where T: Iterator<Item: ?Trait1> {}
//~^ ERROR this relaxed bound is not permitted here

struct S1<T>(T);

impl<T> S1<T> {
    fn f() where T: ?Trait1 {}
    //~^ ERROR this relaxed bound is not permitted here
}

trait Trait5<'a> {}

struct S2<T>(T) where for<'a> T: ?Trait5<'a>;
//~^ ERROR this relaxed bound is not permitted here
//~| ERROR bound modifier `?` can only be applied to default traits like `Sized`

struct S;
impl !Trait2 for S {}
impl Trait1 for S {}
impl Trait3 for S {}

fn main() {
    foo(Box::new(S));
    bar(&S);
}
