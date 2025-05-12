type Foo = Bar;
//~^ ERROR cannot find type `Bar` in this scope

fn check(f: impl FnOnce(Foo), val: Foo) {
    f(val);
}

fn main() {}
