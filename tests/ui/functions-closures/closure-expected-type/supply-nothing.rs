//@ run-pass
fn with_closure<F>(f: F) -> u32
    where F: FnOnce(&u32, &u32) -> u32
{
    f(&22, &44)
}

fn main() {
    let z = with_closure(|x, y| x + y).wrapping_add(1);
    assert_eq!(z, 22 + 44 + 1);
}
