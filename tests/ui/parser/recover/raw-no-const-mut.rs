fn a() {
    let x = &raw 1;
    //~^ ERROR expected one of
}

fn b() {
    [&raw const 1, &raw 2]
    //~^ ERROR expected one of
    //~| ERROR cannot find value `raw` in this scope
    //~| ERROR cannot take address of a temporary
}

fn c() {
    if x == &raw z {}
    //~^ ERROR expected `{`
}

fn d() {
    f(&raw 2);
    //~^ ERROR expected one of
    //~| ERROR cannot find value `raw` in this scope
    //~| ERROR cannot find function `f` in this scope
}

fn e() {
    let x;
    x = &raw 1;
    //~^ ERROR expected one of
}

fn main() {}
