type boxedFn = { theFn: fn () -> uint };

fn createClosure (closedUint: uint) -> boxedFn {
    { theFn: fn@ () -> uint { closedUint } } //~ ERROR mismatched types
}

fn main () {
    let aFn: boxedFn = createClosure(10);

    let myInt: uint = (aFn.theFn)();

    assert myInt == 10;
}