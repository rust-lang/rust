#[derive(Copy, Clone, PartialEq, Debug)]
struct A<'a> {
    x: i32,
    y: &'a i32,
}

fn main() {
    let x = 5;
    let a = A { x: 99, y: &x };
    assert_eq!(Some(a).map(Some), Some(Some(a)));
}
