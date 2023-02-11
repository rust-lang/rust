fn main() {
    let _: f64 = 0..10; //~ ERROR mismatched types
    let _: f64 = 1..; //~ ERROR mismatched types
    let _: f64 = ..10; //~ ERROR mismatched types
    let _: f64 = std::ops::Range { start: 0, end: 1 }; //~ ERROR mismatched types
}
