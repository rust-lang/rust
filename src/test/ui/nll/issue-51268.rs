// ignore-tidy-linelength

struct Bar;

impl Bar {
    fn bar(&mut self, _: impl Fn()) {}
}

struct Foo {
    thing: Bar,
    number: usize,
}

impl Foo {
    fn foo(&mut self) {
        self.thing.bar(|| {
        //~^ ERROR cannot borrow `self.thing` as mutable because it is also borrowed as immutable [E0502]
            &self.number;
        });
    }
}

fn main() {}
