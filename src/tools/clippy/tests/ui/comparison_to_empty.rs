#![warn(clippy::comparison_to_empty)]
#![allow(clippy::borrow_deref_ref, clippy::needless_if, clippy::useless_vec)]
#![feature(let_chains)]

fn main() {
    // Disallow comparisons to empty
    let s = String::new();
    let _ = s == "";
    let _ = s != "";

    let v = vec![0];
    let _ = v == [];
    let _ = v != [];
    if let [] = &*v {}
    let s = [0].as_slice();
    if let [] = s {}
    if let [] = &*s {}
    if let [] = &*s && s == [] {}

    // Allow comparisons to non-empty
    let s = String::new();
    let _ = s == " ";
    let _ = s != " ";

    let v = vec![0];
    let _ = v == [0];
    let _ = v != [0];
    if let [0] = &*v {}
    let s = [0].as_slice();
    if let [0] = s {}
    if let [0] = &*s && s == [0] {}
}
