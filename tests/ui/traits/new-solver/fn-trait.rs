// compile-flags: -Ztrait-solver=next
// check-pass

fn require_fn(_: impl Fn() -> i32) {}

fn f() -> i32 {
    1i32
}

fn main() {
    require_fn(f);
    require_fn(f as fn() -> i32);
}
