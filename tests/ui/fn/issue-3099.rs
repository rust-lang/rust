fn a(x: String) -> String {
    format!("First function with {}", x)
}

fn a(x: String, y: String) -> String { //~ ERROR the name `a` is defined multiple times
    format!("Second function with {} and {}", x, y)
}

fn main() {
    println!("Result: ");
}
