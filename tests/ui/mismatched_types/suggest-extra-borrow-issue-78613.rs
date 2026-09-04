//! Regression test for https://github.com/rust-lang/rust/issues/78613.
//! A method argument that needs one more reference should suggest borrowing it.

fn main() {
    let haystack = [&["A1", "A2"][..], &["B1", "B2"], &["C1", "C2"]];
    let needle: &[&str] = &["D1", "D2"];
    let _ = haystack.contains(needle);
    //~^ ERROR mismatched types
}
