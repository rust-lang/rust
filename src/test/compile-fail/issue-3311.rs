#[legacy_mode]
struct Foo {
    s: &str,
    u: ~()
}

impl Foo {
    fn get_s(&self) -> &self/str {
        self.s
    }
}

fn bar(s: &str, f: fn(Option<Foo>)) {
    f(Some(Foo {s: s, u: ~()}));
}

fn main() {
    do bar(~"testing") |opt| {
        io::println(option::unwrap(opt).get_s()); //~ ERROR illegal borrow:
    };
}
