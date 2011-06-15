

fn main() {
    fn foo(float n) -> float { ret n + 0.12345; }
    let float n = 0.1;
    let float m = foo(n);
    log m;
}