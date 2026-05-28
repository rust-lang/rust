fn a() {
    let x = &raw 1;
    //~^ ERROR expected one of
}

fn b() {
    [&raw const 1, &raw 2]
    //~^ ERROR expected one of
}

fn c() {
    if x == &raw z {}
    //~^ ERROR expected `{`
}

fn d() {
    f(&raw 2);
    //~^ ERROR expected one of
    //~| ERROR cannot find function `f` in this scope
}

fn e() {
    let x;
    x = &raw 1;
    //~^ ERROR expected one of
}

fn main() {}
