fn incr(x: &mut isize) -> bool { *x += 1; assert!((false)); return false; }

pub fn main() {
    let x = 1 == 2 || 3 == 3;
    assert!((x));
    let mut y: isize = 10;
    println!("{}", x || incr(&mut y));
    assert_eq!(y, 10);
    if true && x { assert!((true)); } else { assert!((false)); }
}
