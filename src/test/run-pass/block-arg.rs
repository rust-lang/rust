// Check usage and precedence of block arguments in expressions:
pub fn main() {
    let v = vec![-1.0f64, 0.0, 1.0, 2.0, 3.0];

    // Statement form does not require parentheses:
    for i in &v {
        println!("{}", *i);
    }

}
