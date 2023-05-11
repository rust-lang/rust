// check-pass

fn main() {
    let tuple = (((),),);

    let _ = tuple. 0.0; // OK, whitespace
    let _ = tuple.0. 0; // OK, whitespace

    let _ = tuple./*special cases*/0.0; // OK, comment
}
