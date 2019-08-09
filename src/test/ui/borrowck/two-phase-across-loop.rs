// Test that a borrow which starts as a 2-phase borrow and gets
// carried around a loop winds up conflicting with itself.

struct Foo { x: String }

impl Foo {
    fn get_string(&mut self) -> &str {
        &self.x
    }
}

fn main() {
    let mut foo = Foo { x: format!("Hello, world") };
    let mut strings = vec![];

    loop {
        strings.push(foo.get_string()); //~ ERROR cannot borrow `foo` as mutable
        if strings.len() > 2 { break; }
    }

    println!("{:?}", strings);
}
