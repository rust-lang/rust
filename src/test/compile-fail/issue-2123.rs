use x = m::f; //~ ERROR failed to resolve import

mod m {
    #[legacy_exports];
}

fn main() {
}
