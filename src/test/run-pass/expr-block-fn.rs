fn test_fn() {
    fn ten() -> isize { return 10; }
    let rs = ten;
    assert_eq!(rs(), 10);
}

pub fn main() { test_fn(); }
