// compile-flags: -Z continue-parse-after-error

struct Foo {
    x: isize,
    y: isize,
}

fn main() {
    let f = Foo {
        x: 1,
        y: 2,
    };
    f.x::<isize>;
    //~^ ERROR field expressions may not have generic arguments
    f.x::<>;
    //~^ ERROR field expressions may not have generic arguments
    f.x::();
    //~^ ERROR field expressions may not have generic arguments
}
