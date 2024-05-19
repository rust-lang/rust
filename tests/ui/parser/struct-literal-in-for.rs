struct Foo {
    x: isize,
}

impl Foo {
    fn hi(&self) -> bool {
        true
    }
}

fn main() {
    for x in Foo { //~ ERROR struct literals are not allowed here
        x: 3       //~^ ERROR `bool` is not an iterator
    }.hi() {
        println!("yo");
    }
}
