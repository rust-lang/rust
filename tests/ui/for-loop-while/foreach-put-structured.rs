//@ run-pass


fn pairs<F>(mut it: F) where F: FnMut((isize, isize)) {
    let mut i: isize = 0;
    let mut j: isize = 0;
    while i < 10 { it((i, j)); i += 1; j += i; }
}

pub fn main() {
    let mut i: isize = 10;
    let mut j: isize = 0;
    pairs(|p| {
        let (_0, _1) = p;
        println!("{}", _0);
        println!("{}", _1);
        assert_eq!(_0 + 10, i);
        i += 1;
        j = _1;
    });
    assert_eq!(j, 45);
}
