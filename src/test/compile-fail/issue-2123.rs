// xfail-test
// error-pattern:unresolved import: m::f
use x = m::f;

mod m {
    #[legacy_exports];
}

fn main() {
}
