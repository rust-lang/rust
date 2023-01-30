fn foo(x: u32) -> u32 {
    x * 2
}

fn bar(x: u32) -> u32 {
    x * 3
}

// original example from Issue #102608
fn foobar(n: u32) -> u32 {
    let g = if n % 2 == 0 { &foo } else { &bar };
    //~^ ERROR `if` and `else` have incompatible types
    //~| different fn items have unique types, even if their signatures are the same
    g(n)
}

fn main() {
    assert_eq!(foobar(7), 21);
    assert_eq!(foobar(8), 16);

    // general mismatch of fn item types
    let mut a = foo;
    a = bar;
    //~^ ERROR mismatched types
    //~| expected fn item `fn(_) -> _ {foo}`
    //~| found fn item `fn(_) -> _ {bar}`
    //~| different fn items have unique types, even if their signatures are the same

    // display note even when boxed
    let mut b = Box::new(foo);
    b = Box::new(bar);
    //~^ ERROR mismatched types
    //~| different fn items have unique types, even if their signatures are the same

    // suggest removing reference
    let c: fn(u32) -> u32 = &foo;
    //~^ ERROR mismatched types
    //~| expected fn pointer `fn(u32) -> u32`
    //~| found reference `&fn(u32) -> u32 {foo}`

    // suggest using reference
    let d: &fn(u32) -> u32 = foo;
    //~^ ERROR mismatched types
    //~| expected reference `&fn(u32) -> u32`
    //~| found fn item `fn(u32) -> u32 {foo}`

    // suggest casting with reference
    let e: &fn(u32) -> u32 = &foo;
    //~^ ERROR mismatched types
    //~| expected reference `&fn(u32) -> u32`
    //~| found reference `&fn(u32) -> u32 {foo}`

    // OK
    let mut z: fn(u32) -> u32 = foo as fn(u32) -> u32;
    z = bar;
}
