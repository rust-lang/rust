type Foo = Bar;
//~^ ERROR cannot find type `Bar`

fn check(f: impl FnOnce(Foo), val: Foo) {
    f(val);
}

fn main() {}
