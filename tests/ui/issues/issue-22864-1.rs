// run-pass
pub fn main() {
    struct Fun<F>(F);
    let f = Fun(|x| 3*x);
    let Fun(g) = f;
    println!("{:?}",g(4));
}
