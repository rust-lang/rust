// Test the interaction between rested RPIT and HRTB.

trait Foo<'a> {
    type Assoc;
}

impl Foo<'_> for () {
    type Assoc = ();
}

// Alternative version of `Foo` whose impl uses `'a`.
trait Bar<'a> {
    type Assoc;
}

impl<'a> Bar<'a> for () {
    type Assoc = &'a ();
}

trait Qux<'a> {}

impl Qux<'_> for () {}

// This is not supported.
fn one_hrtb_outlives() -> impl for<'a> Foo<'a, Assoc = impl Sized + 'a> {}
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

// This is not supported.
fn one_hrtb_trait_param() -> impl for<'a> Foo<'a, Assoc = impl Qux<'a>> {}
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

fn one_hrtb_outlives_uses() -> impl for<'a> Bar<'a, Assoc = impl Sized + 'a> {}
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

fn one_hrtb_trait_param_uses() -> impl for<'a> Bar<'a, Assoc = impl Qux<'a>> {}
//~^ ERROR `impl Trait` cannot capture higher-ranked lifetime from outer `impl Trait`

// This should resolve.
fn one_hrtb_mention_fn_trait_param<'b>() -> impl for<'a> Foo<'a, Assoc = impl Qux<'b>> {}

// This should resolve.
fn one_hrtb_mention_fn_outlives<'b>() -> impl for<'a> Foo<'a, Assoc = impl Sized + 'b> {}

// This should resolve.
fn one_hrtb_mention_fn_trait_param_uses<'b>() -> impl for<'a> Bar<'a, Assoc = impl Qux<'b>> {}
//~^ ERROR the trait bound `for<'a> &'a (): Qux<'b>` is not satisfied

// This should resolve.
fn one_hrtb_mention_fn_outlives_uses<'b>() -> impl for<'a> Bar<'a, Assoc = impl Sized + 'b> {}
//~^ ERROR implementation of `Bar` is not general enough
//~| ERROR lifetime may not live long enough

// This should resolve.
fn two_htrb_trait_param() -> impl for<'a> Foo<'a, Assoc = impl for<'b> Qux<'b>> {}

// `'b` is not in scope for the outlives bound.
fn two_htrb_outlives() -> impl for<'a> Foo<'a, Assoc = impl for<'b> Sized + 'b> {}
//~^ ERROR use of undeclared lifetime name `'b` [E0261]

// This should resolve.
fn two_htrb_trait_param_uses() -> impl for<'a> Bar<'a, Assoc = impl for<'b> Qux<'b>> {}
//~^ ERROR: the trait bound `for<'a, 'b> &'a (): Qux<'b>` is not satisfied

// `'b` is not in scope for the outlives bound.
fn two_htrb_outlives_uses() -> impl for<'a> Bar<'a, Assoc = impl for<'b> Sized + 'b> {}
//~^ ERROR use of undeclared lifetime name `'b` [E0261]

fn main() {}
