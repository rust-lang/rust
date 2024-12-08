//@ compile-flags: -Z unpretty=thir-flat
//@ check-pass

// Previously, the constants with `Self::Bar(())` would be `Call`s instead of
// `Adt`s in THIR.

pub enum Foo {
    Bar(()),
}

impl Foo {
    const BAR1: Foo = Foo::Bar(());
    const BAR2: Foo = Self::Bar(());
    const BAR3: Self = Foo::Bar(());
    const BAR4: Self = Self::Bar(());
}

fn main() {}
