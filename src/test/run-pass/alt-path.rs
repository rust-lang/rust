

mod m1 {
    enum foo { foo1, foo2, }
}

fn bar(x: m1::foo) { alt x { m1::foo1 { } m1::foo2 { } } }

fn main() { }
