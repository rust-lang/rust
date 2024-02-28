struct Foo;

trait Bar {}

impl Bar for Foo {}

fn needs_bar<T: Bar>(_: T) {}

fn blah(f: fn() -> Foo) {
    needs_bar(f);
    //~^ ERROR trait `Bar` is not implemented for `fn() -> Foo`
    //~| HELP use parentheses to call this function pointer
}

fn main() {}
