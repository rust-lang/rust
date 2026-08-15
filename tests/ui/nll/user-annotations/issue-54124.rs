fn test<'a>() {
    let _:fn(&()) = |_:&'a ()| {}; //~ ERROR lifetime may not live long enough
    //~^ ERROR lifetime may not live long enough
}

fn main() {
    test();
}
