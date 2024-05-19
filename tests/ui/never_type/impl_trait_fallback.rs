//@ check-pass

fn main() {}

trait T {}
impl T for () {}

fn should_ret_unit() -> impl T {
    panic!()
}
