

fn main() {
    fn foo(n: float) -> float { return n + 0.12345; }
    let n: float = 0.1;
    let m: float = foo(n);
    log(debug, m);
}
