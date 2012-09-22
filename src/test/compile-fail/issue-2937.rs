// error-pattern:failed to resolve imports
use x = m::f;

mod m {
    #[legacy_exports];
}

fn main() {
}
