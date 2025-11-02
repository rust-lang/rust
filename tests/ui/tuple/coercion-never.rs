// This test checks if tuple elements are a coercion site or not.
// Note that the code here is a degenerate case, but you can get similar effects in real code, when
// unifying match arms, for example.
//
// See also coercion-slice.rs
//
//@ check-pass

fn main() {
    let _: ((),) = (loop {},);

    ((),) = (loop {},);

    let x = (loop {},);
    let _: ((),) = x;
}
