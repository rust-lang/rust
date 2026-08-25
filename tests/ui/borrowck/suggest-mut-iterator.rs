//@ run-rustfix
struct Test {
    a: u32
}

impl Test {
    pub fn add(&mut self, value: u32) {
        self.a += value;
    }

    pub fn print_value(&self) {
        println!("Value of a is: {}", self.a);
    }
}

fn main() {
    let mut tests = Vec::new();
    for i in 0..=10 {
        tests.push(Test {a: i});
    }
    for test in &tests {
        test.add(2); //~ ERROR cannot borrow `*test` as mutable, as it is behind a `&` reference
    }
    for test in &mut tests {
        test.add(2);
    }
    for test in &tests {
        test.print_value();
    }
}
