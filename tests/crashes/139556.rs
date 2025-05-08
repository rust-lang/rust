//@ known-bug: #139556

trait T {}

type Alias<'a> = impl T;

struct S;
impl<'a> T for &'a S {}

#[define_opaque(Alias)]
fn with_positive(fun: impl Fn(Alias<'_>)) {
    with_positive(|&n| ());
}
