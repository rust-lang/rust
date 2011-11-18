// error-pattern:expected fn() but found fn(++int)

fn main() {
    fn f() { }
    fn g(i: int) { }
    let x = f == g;
}
