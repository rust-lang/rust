

fn test_fn() {
    type t = extern fn() -> int;
    fn ten() -> int { ret 10; }
    let rs: t = { ten };
    assert (rs() == 10);
}

fn main() { test_fn(); }
