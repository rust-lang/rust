// fn foo() -> String {
//    String::new()
// }

fn test(s: &str) {
    println!("{}", s);
}

fn test2(s: String) {
    println!("{}", s);
}

fn main() {
    let x = foo(); //~ERROR cannot find function `foo`
    test(&x);
    test2(x); // Does not complain about `x` being a `&str`.
}
