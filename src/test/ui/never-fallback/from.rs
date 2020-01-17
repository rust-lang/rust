// check-pass
// Tests that we correctly infer types to !
#![feature(never_type)]
#![feature(never_type_fallback)]

struct E;
impl From<!> for E {
    fn from(x: !) -> E { x }
}
fn foo(never: !) {
    <E as From<_>>::from(never);
}

fn main() {}
