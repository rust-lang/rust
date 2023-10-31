// compile-flags: -Ztrait-solver=next
// check-pass

const fn foo() {
    let mut x = [1, 2, 3];
    // We need to fix up `<<[i32; 3] as Index<usize>>::Output as AddAssign>`
    // to be treated like a built-in operation.
    x[1] += 5;
}

fn main() {}
