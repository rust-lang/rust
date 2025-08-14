//@ revisions: become return
//@ [become] run-pass
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

// This is an identity function (`|x| x`), but implemented using recursion.
// Each step we increment accumulator and decrement the number.
//
// With normal calls this fails compilation because of the recursion limit,
// but with tail calls/`become` we don't grow the stack/spend recursion limit
// so this should compile.
const fn rec_id(n: u32) -> u32 {
    const fn inner(acc: u32, n: u32) -> u32 {
        match n {
            0 => acc,
            #[cfg(r#become)] _ => become inner(acc + 1, n - 1),
            #[cfg(r#return)] _ => return inner(acc + 1, n - 1),
        }
    }

    inner(0, n)
}

// Some relatively big number that is higher than recursion limit
const ORIGINAL: u32 = 12345;
// Original number, but with identity function applied
// (this is the same, but requires execution of the recursion)
const ID_ED: u32 = rec_id(ORIGINAL); //[return]~ ERROR: reached the configured maximum number of stack frames
// Assert to make absolutely sure the computation actually happens
const ASSERT: () = assert!(ORIGINAL == ID_ED);

fn main() {
    let _ = ASSERT;
}
