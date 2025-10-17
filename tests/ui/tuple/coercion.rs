// This test checks if tuple elements are a coercion site or not.
// Note that the code here is a degenerate case, but you can get similar effects in real code, when
// unifying match arms, for example.

fn main() {
    let _: ((),) = (loop {},);

    ((),) = (loop {},);

    let x = (loop {},);
    let _: ((),) = x;

    let _: (&[u8],) = (&[],);

    // This one can't work without a redesign on the coercion system.
    // We currently only eagerly add never-to-any coercions, not any others.
    // Thus, because we don't have an expectation when typechecking `&[]`,
    // we don't add a coercion => this doesn't work.
    let y = (&[],);
    let _: (&[u8],) = y; //~ error: mismatched types
}
