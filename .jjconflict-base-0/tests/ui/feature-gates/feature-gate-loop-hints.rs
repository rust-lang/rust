fn main() {
    #[rustc_unroll] //~ ERROR the `rustc_unroll` attribute is an experimental feature
    for _ in 0..10 {}
}
