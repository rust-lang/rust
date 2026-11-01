 fn main() {
    |x: usize| [0; x];  //~ ERROR attempt to use a non-constant value in a constant [E0435]
    // (note the space before "fn")
}
