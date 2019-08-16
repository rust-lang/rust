// run-pass
// Test direct calls to extern fns.


extern fn f(x: usize) -> usize { x * 2 }

pub fn main() {
    let x = f(22);
    assert_eq!(x, 44);
}
