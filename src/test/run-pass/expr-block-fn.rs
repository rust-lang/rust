

fn test_fn() {
    type t = fn@() -> int;
    fn ten() -> int { return 10; }
    let rs: t = { ten };
    //assert (rs() == 10);
}

fn main() { test_fn(); }
