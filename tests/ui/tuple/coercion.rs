// This test checks if tuple elements are a coercion site or not.
// Note that the code here is a degenerate case, but you can get similar effects in real code, when
// unifying match arms, for example.

fn main() {
    let _: ((),) = (loop {},);

    ((),) = (loop {},); //~ error: mismatched types

    let x = (loop {},);
    let _: ((),) = x; //~ error: mismatched types

    let _: (&[u8],) = (&[],);

    let y = (&[],);
    let _: (&[u8],) = y; //~ error: mismatched types
}
