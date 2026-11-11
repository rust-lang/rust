// issue#126764

struct S;

trait T {
    fn f();
}
impl T for S；
//~^ ERROR: unknown start of token
//~| ERROR: expected `{}`
//~| ERROR: not all trait items implemented, missing: `f`

fn main() {}
