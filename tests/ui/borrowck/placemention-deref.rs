enum Thing {
    Wut,
}

struct A(Box<A>);

fn main() {
    let x: &'static String;
    let _ = *x;
    //~^ ERROR used binding `x` isn't initialized

    let x: &Thing;
    let Thing::Wut = *x;
    //~^ ERROR used binding `x` isn't initialized

    let x = Box::new(1);
    drop(x);
    _ = *x;
    //~^ ERROR use of moved value: `x`

    let x: A;
    _ = x.0.0.0.0.0;
    //~^ ERROR used binding `x` isn't initialized

    let x: A;
    let _ = x.0.0;
    //~^ ERROR used binding `x` isn't initialized

    let x = Box::new(Box::new(Box::new(Box::new(Box::new(Box::new(1))))));
    drop(x);
    _ = ******x;
    //~^ ERROR use of moved value: `x`

    let x = Box::new(Box::new(Box::new(0)));
    drop(**x);
    _ = ***x;
    //~^ ERROR use of moved value: `**x`

    let x = Box::new(Box::new(Box::new(0)));
    drop(**x);
    _ = **x; // OK
}
