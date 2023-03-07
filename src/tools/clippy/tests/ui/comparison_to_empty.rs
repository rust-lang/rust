// run-rustfix

#![warn(clippy::comparison_to_empty)]

fn main() {
    // Disallow comparisons to empty
    let s = String::new();
    let _ = s == "";
    let _ = s != "";

    let v = vec![0];
    let _ = v == [];
    let _ = v != [];

    // Allow comparisons to non-empty
    let s = String::new();
    let _ = s == " ";
    let _ = s != " ";

    let v = vec![0];
    let _ = v == [0];
    let _ = v != [0];
}
