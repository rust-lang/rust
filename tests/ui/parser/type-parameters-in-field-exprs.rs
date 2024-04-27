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
    //~^ ERROR field expressions cannot have generic arguments
    f.x::<>;
    //~^ ERROR field expressions cannot have generic arguments
    f.x::();
    //~^ ERROR field expressions cannot have generic arguments
}
