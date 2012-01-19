

mod m1 {
    tag foo { foo1; foo2; }
}

fn bar(x: m1::foo) { alt x { m1::foo1 { } } }

fn main() { }
