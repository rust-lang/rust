// error-pattern:import

mod a {
    #[legacy_exports];
    import b::x;
    export x;
}

mod b {
    #[legacy_exports];
    import a::x;
    export x;

    fn main() { let y = x; }
}
