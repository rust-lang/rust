//@ check-pass

// Ensure that we don't ICE when an alias that has escaping bound vars is
// required to be live. This is because the code that allows us to deduce an
// appropriate outlives bound for a given alias type (in this test, a
// projection) does not handle aliases with escaping bound vars.
// See <https://github.com/rust-lang/rust/issues/117455>.

trait Foo {
    type Assoc<'a, 'b>: 'static;
}

struct MentionsLifetimeAndType<'a, T>(&'a (), T);

fn foo<'a, 'b, T: Foo>(_: <T as Foo>::Assoc<'a, 'b>) {}

fn test<'b, T: Foo>() {
    let y: MentionsLifetimeAndType<'_, for<'a> fn(<T as Foo>::Assoc<'a, 'b>)> =
        MentionsLifetimeAndType(&(), foo);
}

fn main() {}
