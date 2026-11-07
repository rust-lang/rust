//@ edition:2021
// gate-test-anonymous_lifetime_in_impl_trait
// Verify the behaviour of `feature(anonymous_lifetime_in_impl_trait)`.

mod elided {
    fn f(_: impl Iterator<Item = &()>) {}
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable

    fn g(mut x: impl Iterator<Item = &()>) -> Option<&()> { x.next() }
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable
    //~| ERROR missing lifetime specifier

    // Anonymous lifetimes in async fn are already allowed.
    // This is understood as `fn foo<'_1>(_: impl Iterator<Item = &'_1 ()>) {}`.
    async fn h(_: impl Iterator<Item = &()>) {}

    // Anonymous lifetimes in async fn are already allowed.
    // But that lifetime does not participate in resolution.
    async fn i(mut x: impl Iterator<Item = &()>) -> Option<&()> { x.next() }
    //~^ ERROR missing lifetime specifier
}

mod underscore {
    fn f(_: impl Iterator<Item = &'_ ()>) {}
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable

    fn g(mut x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()> { x.next() }
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable
    //~| ERROR missing lifetime specifier

    // Anonymous lifetimes in async fn are already allowed.
    // This is understood as `fn foo<'_1>(_: impl Iterator<Item = &'_1 ()>) {}`.
    async fn h(_: impl Iterator<Item = &'_ ()>) {}

    // Anonymous lifetimes in async fn are already allowed.
    // But that lifetime does not participate in resolution.
    async fn i(mut x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()> { x.next() }
    //~^ ERROR missing lifetime specifier
}

mod alone_in_path {
    trait Foo<'a> { fn next(&mut self) -> Option<&'a ()>; }

    fn f(_: impl Foo) {}
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable

    fn g(mut x: impl Foo) -> Option<&()> { x.next() }
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable
    //~| ERROR missing lifetime specifier
}

mod alone_in_path2 {
    trait Foo<'a> { fn next(&mut self) -> Option<&'a ()>; }

    fn f(_: impl Foo<>) {}
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable

    fn g(mut x: impl Foo<>) -> Option<&()> { x.next() }
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable
    //~| ERROR missing lifetime specifier
}

mod in_path {
    trait Foo<'a, T> { fn next(&mut self) -> Option<&'a T>; }

    fn f(_: impl Foo<()>) {}
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable

    fn g(mut x: impl Foo<()>) -> Option<&()> { x.next() }
    //~^ ERROR anonymous lifetimes in `impl Trait` are unstable
    //~| ERROR missing lifetime specifier
}

// This must not err, as the `&` actually resolves to `'a`.
fn resolved_anonymous<'a, T: 'a>(f: impl Fn(&'a str) -> &T) {
    f("f");
}

fn main() {}
