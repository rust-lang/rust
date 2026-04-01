//@ run-pass

struct Foo {
    x: u32
}

impl Foo {
    fn twiddle(&mut self) -> &mut Self { self }
    fn twaddle(&mut self) -> &mut Self { self }
    fn emit(&mut self) {
        self.x += 1;
    }
}

fn main() {
    let mut foo = Foo { x: 0 };
    match 22 {
        22 => &mut foo,
        44 => foo.twiddle(),
        _ => foo.twaddle(),
    }.emit();
}
