// Pre-existing behavior has been to reject patterns with consts
// denoting non-empty arrays of non-`Eq` types, but *accept* empty
// arrays of such types.
//
// See rust-lang/rust#62336.

//@ run-pass

#[derive(PartialEq, Debug)]
struct B(i32);

fn main() {
    const FOO: [B; 0] = [];
    match [] {
        FOO => { }
    }
}
