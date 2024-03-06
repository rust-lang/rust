//@ check-pass

fn main() {
    let tuple = (((),),);

    let () = tuple. 0.0; // OK, whitespace
    let () = tuple.0. 0; // OK, whitespace

    let () = tuple./*special cases*/0.0; // OK, comment
}
