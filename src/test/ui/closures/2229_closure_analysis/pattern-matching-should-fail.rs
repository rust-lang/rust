#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete
//~| `#[warn(incomplete_features)]` on by default
//~| see issue #53488 <https://github.com/rust-lang/rust/issues/53488>
#![feature(never_type)]

// Should fake read the discriminant and throw an error
fn test1() {
    let x: !;
    let c1 = || match x { };
    //~^ ERROR: use of possibly-uninitialized variable: `x`
}

// Should fake read the discriminant and throw an error
fn test2() {
    let x: !;
    let c2 = || match x { _ => () };
    //~^ ERROR: borrow of possibly-uninitialized variable: `x`
}

// Testing single variant patterns
enum SingleVariant {
    Points(u32)
}

// Should fake read the discriminant and throw an error
fn test3() {
    let variant: !;
    let c = || {
    //~^ ERROR: borrow of possibly-uninitialized variable: `variant`
        match variant {
            SingleVariant::Points(_) => {}
        }
    };
    c();
}

// Should fake read the discriminant and throw an error
fn test4() {
    let variant: !;
    let c = || {
    //~^ ERROR: borrow of possibly-uninitialized variable: `variant`
        match variant {
            SingleVariant::Points(a) => {
                println!("{:?}", a);
            }
        }
    };
    c();
}

fn test5() {
    let t: !;
    let g: !;

    let a = || {
        match g { };
        //~^ ERROR: use of possibly-uninitialized variable: `g`
        let c = ||  {
            match t { };
            //~^ ERROR: use of possibly-uninitialized variable: `t`
        };

        c();
    };

}

// Should fake read the discriminant and throw an error
fn test6() {
    let x: u8;
    let c1 = || match x { };
    //~^ ERROR: use of possibly-uninitialized variable: `x`
    //~| ERROR: non-exhaustive patterns: type `u8` is non-empty
}

fn main() {
    test1();
    test2();
    test3();
    test4();
    test5();
    test6();
}
