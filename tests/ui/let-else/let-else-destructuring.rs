#[derive(Debug)]
enum Foo {
    Done,
    Nested(Option<&'static Foo>),
}

fn walk(mut value: &Foo) {
    loop {
        println!("{:?}", value);
        &Foo::Nested(Some(value)) = value else { break }; //~ ERROR invalid left-hand side of assignment
        //~^ERROR <assignment> ... else { ... } is not allowed
    }
}

fn main() {
    walk(&Foo::Done);
}
