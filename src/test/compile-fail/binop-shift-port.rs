// xfail-stage0
// error-pattern:>> cannot be applied to type `port[int]`

fn main() {
    let p1: port[int] = port();
    let p2: port[int] = port();
    let x = p1 >> p2;
}