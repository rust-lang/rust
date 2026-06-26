//@ check-pass

// Test that `ref` patterns may be used on both sides
// of an `@` pattern according to NLL borrowck.

fn main() {
    struct U; // Not copy!

    // Promotion:
    let ref a @ ref b = U;
    let _: &U = a;
    let _: &U = b;

    // Prevent promotion:
    fn u() -> U { U }

    let ref a @ ref b = u();
    let _: &U = a;
    let _: &U = b;

    let ref a @ (ref b, [ref c, ref d]) = (u(), [u(), u()]);
    let _: &(U, [U; 2]) = a;
    let _: &U = b;
    let _: &U = c;
    let _: &U = d;

    fn f1(ref a @ (ref b, [ref c, ref mid @ .., ref d]): (U, [U; 4])) {}

    let a @ (b, [c, d]) = &(u(), [u(), u()]);
    let _: &(U, [U; 2]) = a;
    let _: &U = b;
    let _: &U = c;
    let _: &U = d;

    let ref a @ &ref b = &u();
    let _: &&U = a;
    let _: &U = b;

    match Ok(u()) {
        ref a @ Ok(ref b) | ref a @ Err(ref b) => {
            let _: &Result<U, U> = a;
            let _: &U = b;
        }
    }
}
