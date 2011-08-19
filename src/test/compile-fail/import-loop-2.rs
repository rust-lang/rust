// error-pattern:cyclic import

mod a {
    import b::x;
    export x;
}

mod b {
    import a::x;
    export x;

    fn main() { let y = x; }
}
