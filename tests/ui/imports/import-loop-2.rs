// error-pattern:import

mod a {
    pub use b::x;
}

mod b {
    pub use a::x;

    fn main() { let y = x; }
}

fn main() {}
