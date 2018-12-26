struct X {
    x: String,
}

impl Drop for X {
    fn drop(&mut self) {
        println!("value: {}", self.x);
    }
}

fn unwrap(x: X) -> String {
    let X { x: y } = x; //~ ERROR cannot move out of type
    y
}

fn main() {
    let x = X { x: "hello".to_string() };
    let y = unwrap(x);
    println!("contents: {}", y);
}
