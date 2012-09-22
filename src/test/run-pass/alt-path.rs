

mod m1 {
    #[legacy_exports];
    enum foo { foo1, foo2, }
}

fn bar(x: m1::foo) { match x { m1::foo1 => { } m1::foo2 => { } } }

fn main() { }
