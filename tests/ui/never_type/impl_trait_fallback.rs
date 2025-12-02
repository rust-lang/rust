//@ check-pass
#![expect(dependency_on_unit_never_type_fallback)]

fn main() {}

trait T {}
impl T for () {}

fn should_ret_unit() -> impl T {
    panic!()
}
