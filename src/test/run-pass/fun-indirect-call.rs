fn f() -> isize { return 42; }

pub fn main() {
    let g: fn() -> isize = f;
    let i: isize = g();
    assert_eq!(i, 42);
}
