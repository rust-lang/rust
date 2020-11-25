// check-pass

// https://github.com/rust-lang/rust/issues/79152
const fn foo() {
    let mut array = [[0; 1]; 1];
    array[0][0] = 1;
}

pub fn main() {
    foo()
}
