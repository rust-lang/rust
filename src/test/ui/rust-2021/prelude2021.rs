// check-pass
// edition:2021
// compile-flags: -Zunstable-options

fn main() {
    let _: u16 = 123i32.try_into().unwrap();
}
