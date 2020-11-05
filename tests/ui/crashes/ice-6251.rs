// originally from glacier/fixed/77329.rs
// assertion failed: `(left == right) ; different DefIds

fn bug<T>() -> impl Iterator<Item = [(); { |x: [u8]| x }]> {
    std::iter::empty()
}
