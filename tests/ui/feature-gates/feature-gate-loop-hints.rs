fn main() {
    #[unroll] //~ ERROR the `unroll` attribute is an experimental feature
    for _ in 0..10 {}
}
