

fn test_fn() {
    type t = fn() -> int ;

    fn ten() -> int { ret 10; }
    let t rs = { ten };
    assert (rs() == 10);
}

fn main() { test_fn(); }