fn foo(_x: u32, a: impl Into<String>, _y: u32, b: impl Into<String>) {
    println!("fox: a={}, b={}", a.into(), b.into());
}

fn main() {
    let bar: String = "bar".to_string();
    let baz: &str = "baz";

    for (a, b) in &[(&bar, baz)] {
        let a: &String = a;
        foo(42, a, 43, b); //~ ERROR: the trait bound `String: From<&&str>` is not satisfied [E0277]
    }
}
