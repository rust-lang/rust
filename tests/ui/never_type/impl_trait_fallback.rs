//@ check-pass

fn main() {}

trait T {}
impl T for () {}

fn should_ret_unit() -> impl T {
    //~^ warn: this function depends on never type fallback being `()`
    //~| warn: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    panic!()
}
