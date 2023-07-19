// Make sure we blame the furthest statment in lifetime errors.

fn id<X>(x: X) -> X { x }
fn relate<X>(_: &X, _: &X) {}
struct Inv<'a>(*mut &'a u8);

fn test_static(a: Inv<'_>, b: Inv<'static>) {
    let a = id(a);
    let b = id(b);
    relate(&a, &b);
    //~^ ERROR
}

fn test1(a: Inv<'_>, b: Inv<'_>) {
    let a = id(a);
    let b = id(b);
    relate(&a, &b);
    //~^ ERROR
    //~| ERROR
}

fn test2(cond: bool, a: Inv<'_>, b: Inv<'_>) {
    let mut x = None::<Inv<'_>>;
    let mut y = None::<Inv<'_>>;
    if cond {
        relate(&x, &y);
    } else {
        x.replace(a);
        y.replace(b);
        //~^ ERROR
        //~| ERROR
    }
}

fn test3(cond: bool, a: Inv<'_>, b: Inv<'_>) {
    let mut x = None::<Inv<'_>>;
    let mut y = None::<Inv<'_>>;
    if cond {
        x.replace(a);
        y.replace(b);
    } else {
        relate(&x, &y);
        //~^ ERROR
        //~| ERROR
    }
}

fn main() {}
