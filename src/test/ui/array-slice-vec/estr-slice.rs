// run-pass


pub fn main() {
    let x = "hello";
    let v = "hello";
    let y : &str = "there";

    println!("{}", x);
    println!("{}", y);

    assert_eq!(x.as_bytes()[0], 'h' as u8);
    assert_eq!(x.as_bytes()[4], 'o' as u8);

    let z : &str = "thing";
    assert_eq!(v, x);
    assert_ne!(x, z);

    let a = "aaaa";
    let b = "bbbb";

    let c = "cccc";
    let cc = "ccccc";

    println!("{}", a);

    assert!(a < b);
    assert!(a <= b);
    assert_ne!(a, b);
    assert!(b >= a);
    assert!(b > a);

    println!("{}", b);

    assert!(a < c);
    assert!(a <= c);
    assert_ne!(a, c);
    assert!(c >= a);
    assert!(c > a);

    println!("{}", c);

    assert!(c < cc);
    assert!(c <= cc);
    assert_ne!(c, cc);
    assert!(cc >= c);
    assert!(cc > c);

    println!("{}", cc);
}
