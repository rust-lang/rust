struct Test {
    a: isize,
    b: Option<Box<Test>>,
}

impl Drop for Test {
    fn drop(&mut self) {
        println!("Dropping {}", self.a);
    }
}

fn stuff() {
    let mut t = Test { a: 1, b: None};
    let mut u = Test { a: 2, b: Some(Box::new(t))};
    t.b = Some(Box::new(u));
    //~^ ERROR assign of moved value: `t`
    println!("done");
}

fn main() {
    stuff();
    println!("Hello, world!")
}
