//! Regression test for https://github.com/rust-lang/rust/issues/76112.
//! A closure argument used as an index should be inferred from a later call to the closure.

fn main() {
    let array: [i64; 1] = [0];
    let get = |index| array[index].pow(1);
    //~^ ERROR type annotations needed

    let value: i64 = get(0);
    assert_eq!(value, 0);
}
