// Test for issue #50381: non-lifetime passed to :lifetime.

macro_rules! m { ($x:lifetime) => { } }

fn main() {
    m!(a);
    //~^ ERROR no rules expected `a`
}
