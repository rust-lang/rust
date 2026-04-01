//@ edition:2021

#![feature(never_type)]

// Should fake read the discriminant and throw an error
fn test1() {
    let x: !;
    let c1 = || match x { };
    //~^ ERROR E0381
}

// Should fake read the discriminant and throw an error
fn test2() {
    let x: !;
    let c2 = || match x { _ => () };
    //~^ ERROR E0381
}

// Testing single variant patterns
enum SingleVariant {
    Points(u32)
}

// Should fake read the discriminant and throw an error
fn test3() {
    let variant: !;
    let c = || {
    //~^ ERROR E0381
        match variant {
            SingleVariant::Points(_) => {}
        }
    };
    c();
}

// Should fake read the discriminant and throw an error
fn test4() {
    let variant: !;
    let c = || { //~ ERROR E0381
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
        match g { }; //~ ERROR E0381
        let c = ||  {
            match t { }; //~ ERROR E0381
        };

        c();
    };

}

// Should fake read the discriminant and throw an error
fn test6() {
    let x: u8;
    let c1 = || match x { };
    //~^ ERROR E0381
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
