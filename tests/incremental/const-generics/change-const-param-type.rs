//@ revisions: rpass1 rpass2 rpass3
//@ compile-flags: -Zincremental-ignore-spans

enum Foo<const N: usize> {
    Variant,
    Variant2(),
    Variant3 {},
}

impl Foo<1> {
    fn foo<const N: usize>(&self) -> [(); N] { [(); N] }
}

impl Foo<2> {
    fn foo<const N: u32>(self) -> usize { N as usize }
}

struct Bar<const N: usize>;
struct Bar2<const N: usize>();
struct Bar3<const N: usize> {}

#[cfg(rpass1)]
struct ChangingStruct<const N: usize>;

#[cfg(any(rpass2, rpass3))]
struct ChangingStruct<const N: u32>;

struct S;

impl S {
    #[cfg(rpass1)]
    fn changing_method<const N: usize>(self) {}

    #[cfg(any(rpass2, rpass3))]
    fn changing_method<const N: u32>(self) {}
}

// We want to verify that all goes well when the value of the const argument change.
// To avoid modifying `main`'s HIR, we use a separate constant, and use `{ FOO_ARG + 1 }`
// inside the body to keep having an `AnonConst` to compute.
const FOO_ARG: usize = if cfg!(rpass2) { 1 } else { 0 };

fn main() {
    let foo = Foo::Variant::<{ FOO_ARG + 1 }>;
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let foo = Foo::Variant2::<{ FOO_ARG + 1 }>();
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let foo = Foo::Variant3::<{ FOO_ARG + 1 }> {};
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let foo = Foo::<{ FOO_ARG + 1 }>::Variant;
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let foo = Foo::<{ FOO_ARG + 1 }>::Variant2();
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let foo = Foo::<{ FOO_ARG + 1 }>::Variant3 {};
    foo.foo::<{ if cfg!(rpass3) { 3 } else { 4 } }>();

    let _ = Bar::<{ FOO_ARG + 1 }>;
    let _ = Bar2::<{ FOO_ARG + 1 }>();
    let _ = Bar3::<{ FOO_ARG + 1 }> {};

    let _ = ChangingStruct::<{ 5 }>;
    let _ = S.changing_method::<{ 5 }>();
}
