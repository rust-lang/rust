//@ run-pass
//@ compile-flags: -O

// Regression test for issue #80309

pub fn zero(x: usize) -> usize {
    std::ptr::null::<i8>().wrapping_add(x) as usize - x
}
pub fn qux(x: &[i8]) -> i8 {
    x[zero(x.as_ptr() as usize)]
}

fn main() {
    let z = vec![42, 43];
    println!("{}", qux(&z));
}
