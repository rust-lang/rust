fn foo(c1: bool, c2: bool, x: u64) -> u64 {
    let r = if c1 { x + 3 } else { x + 4 };

    let r = if c2 { r - 1 } else { r - 2 };
    assert!(r > x);
    r
}
// spec foo {
//     ensures r > x;
// }

fn main() {
    // println!("{}", foo(false, true, 13))
    foo(false, true, 13);
}
