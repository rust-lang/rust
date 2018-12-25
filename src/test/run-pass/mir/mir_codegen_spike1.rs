// run-pass
// A simple spike test for MIR version of codegen.

fn sum(x: i32, y: i32) -> i32 {
    x + y
}

fn main() {
    let x = sum(22, 44);
    assert_eq!(x, 66);
    println!("sum()={:?}", x);
}
