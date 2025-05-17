#![warn(clippy::comparison_to_empty)]
#![allow(clippy::borrow_deref_ref, clippy::needless_if, clippy::useless_vec)]

fn main() {
    // Disallow comparisons to empty
    let s = String::new();
    let _ = s == "";
    //~^ comparison_to_empty
    let _ = s != "";
    //~^ comparison_to_empty

    let v = vec![0];
    let _ = v == [];
    //~^ comparison_to_empty
    let _ = v != [];
    //~^ comparison_to_empty
    if let [] = &*v {}
    //~^ comparison_to_empty
    let s = [0].as_slice();
    if let [] = s {}
    //~^ comparison_to_empty
    if let [] = &*s {}
    //~^ comparison_to_empty
    if let [] = &*s
    //~^ comparison_to_empty
        && s == []
    //~^ comparison_to_empty
    {}

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
    if let [0] = &*s
        && s == [0]
    {}

    // Also lint the `PartialEq` methods
    let s = String::new();
    let _ = s.eq("");
    //~^ comparison_to_empty
    let _ = s.ne("");
    //~^ comparison_to_empty
    let v = vec![0];
    let _ = v.eq(&[]);
    //~^ comparison_to_empty
    let _ = v.ne(&[]);
    //~^ comparison_to_empty
}
