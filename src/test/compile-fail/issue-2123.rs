// xfail-test
// error-pattern:unresolved import: m::f
use x = m::f;

mod m {
}

fn main() {
}
