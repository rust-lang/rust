//@ run-pass
// Test that the call operator autoderefs when calling a bounded type parameter.

fn call_with_2<F>(x: &mut F) -> isize
    where F : FnMut(isize) -> isize
{
    x(2) // look ma, no `*`
}

pub fn main() {
    let z = call_with_2(&mut |x| x - 22);
    assert_eq!(z, -20);
}
