fn f() -> ~int {
    ~100
}

fn main() {
    assert f() == ~100;
}