//@ check-pass

// Ensure that `default_field_values` and `const_default` interact properly.

#![feature(const_default)]
#![feature(const_trait_impl)]
#![feature(default_field_values)]
#![feature(derive_const)]

#[derive(PartialEq, Eq, Debug)]
#[derive_const(Default)]
struct S {
    r: Option<String> = <Option<_> as Default>::default(),
    s: String = String::default(),
    o: Option<String> = Option::<String>::default(),
    p: std::marker::PhantomData<()> = std::marker::PhantomData::default(),
    q: Option<String> = <Option<String> as Default>::default(),
    t: Option<String> = Option::default(),
    v: Option<String> = const { Option::default() },
}

const _: S = S { .. };
const _: S = const { S { .. } };
const _: S = S::default();
const _: S = const { S::default() };

fn main() {
    let s = S { .. };
    assert_eq!(s.r, None);
    assert_eq!(&s.s, "");
    assert_eq!(s.o, None);
    assert_eq!(s.p, std::marker::PhantomData);
    assert_eq!(s.q, None);
    assert_eq!(s.t, None);
    assert_eq!(s.v, None);
    assert_eq!(s, S::default());
}
