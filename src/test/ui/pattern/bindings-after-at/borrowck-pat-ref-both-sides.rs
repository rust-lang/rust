// check-pass

// Test that `ref` patterns may be used on both sides
// of an `@` pattern according to NLL borrowck.

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

fn main() {
    struct U; // Not copy!

    let ref a @ ref b = U;
    let _: &U = a;
    let _: &U = b;

    let ref a @ (ref b, [ref c, ref d]) = (U, [U, U]);
    let _: &(U, [U; 2]) = a;
    let _: &U = b;
    let _: &U = c;
    let _: &U = d;

    let a @ (b, [c, d]) = &(U, [U, U]);
    let _: &(U, [U; 2]) = a;
    let _: &U = b;
    let _: &U = c;
    let _: &U = d;

    let ref a @ &ref b = &U;
    let _: &&U = a;
    let _: &U = b;

    match Ok(U) {
        ref a @ Ok(ref b) | ref a @ Err(ref b) => {
            let _: &Result<U, U> = a;
            let _: &U = b;
        }
    }
}
