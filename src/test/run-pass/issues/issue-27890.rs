// run-pass
static PLUS_ONE: &'static (Fn(i32) -> i32 + Sync) = (&|x: i32| { x + 1 })
    as &'static (Fn(i32) -> i32 + Sync);

fn main() {
    assert_eq!(PLUS_ONE(2), 3);
}
