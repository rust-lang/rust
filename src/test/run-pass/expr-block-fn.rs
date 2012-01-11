

fn test_fn() {
    type t = native fn() -> int;
    fn ten() -> int { ret 10; }
    let rs: t = { ten };
    assert (rs() == 10);
}

fn main() { test_fn(); }
