// error-pattern:woe
fn f(a: isize) {
    println!("{}", a);
}

fn main() {
    f(panic!("woe"));
}
