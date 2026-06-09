//@ dont-require-annotations: NOTE

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
    //~| NOTE different fn items have unique types, even if their signatures are the same
    g(n)
}

fn main() {
    assert_eq!(foobar(7), 21);
    assert_eq!(foobar(8), 16);

    // general mismatch of fn item types
    let mut a = foo;
    a = bar;
    //~^ ERROR mismatched types
    //~| NOTE expected fn item `fn(_) -> _ {foo}`
    //~| NOTE found fn item `fn(_) -> _ {bar}`
    //~| NOTE different fn items have unique types, even if their signatures are the same

    // display note even when boxed
    let mut b = Box::new(foo);
    b = Box::new(bar);
    //~^ ERROR mismatched types
    //~| NOTE different fn items have unique types, even if their signatures are the same

    // suggest removing reference
    let c: fn(u32) -> u32 = &foo;
    //~^ ERROR mismatched types
    //~| NOTE expected fn pointer `fn(_) -> _`
    //~| NOTE found reference `&fn(_) -> _ {foo}`

    // suggest using reference
    let d: &fn(u32) -> u32 = foo;
    //~^ ERROR mismatched types
    //~| NOTE expected reference `&fn(_) -> _`
    //~| NOTE found fn item `fn(_) -> _ {foo}`

    // suggest casting with reference
    let e: &fn(u32) -> u32 = &foo;
    //~^ ERROR mismatched types
    //~| NOTE expected reference `&fn(_) -> _`
    //~| NOTE found reference `&fn(_) -> _ {foo}`

    // OK
    let mut z: fn(u32) -> u32 = foo as fn(u32) -> u32;
    z = bar;
}
