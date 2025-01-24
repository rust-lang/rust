// Note: It is no longer true that both `Eq` and `PartialEq` must the derived, only the later.

#[derive(Eq)]
struct Foo {
    x: u32
}

impl PartialEq for Foo {
    fn eq(&self, _: &Foo) -> bool {
        false // ha ha!
    }
}

const FOO: Foo = Foo { x: 0 };

fn main() {
    let y = Foo { x: 1 };
    match y {
        FOO => { }
        //~^ ERROR constant of non-structural type `Foo` in a pattern
        _ => { }
    }
}
