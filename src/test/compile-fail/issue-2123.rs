// xfail-test
// error-pattern:unresolved import: m::f
import x = m::f;

mod m {
}

fn main() {
}
