//@compile-flags: -Zmiri-retag-fields
fn main() {
    let array = [(); usize::MAX];
    drop(array); // Pass the array to a function, retagging its fields
}
