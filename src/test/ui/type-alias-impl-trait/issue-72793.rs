// build-pass

// Regression test for #72793.
// FIXME: This still shows ICE with `-Zmir-opt-level=2`.

#![feature(type_alias_impl_trait)]

trait T { type Item; }

type Alias<'a> = impl T<Item = &'a ()>;

struct S;
impl<'a> T for &'a S {
    type Item = &'a ();
}

fn filter_positive<'a>() -> Alias<'a> {
    &S
}

fn with_positive(fun: impl Fn(Alias<'_>)) {
    fun(filter_positive());
}

fn main() {
    with_positive(|_| ());
}
