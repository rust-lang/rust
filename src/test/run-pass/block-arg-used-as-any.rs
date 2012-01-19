fn call_any(f: fn() -> uint) -> uint {
    ret f();
}

fn main() {
    let x_r = call_any {|| 22u };
    assert x_r == 22u;
}
