// run-pass


pub fn main() {
    fn foo(n: f64) -> f64 { return n + 0.12345; }
    let n: f64 = 0.1;
    let m: f64 = foo(n);
    println!("{}", m);
}
