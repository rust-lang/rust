#![allow(deprecated)]

enum Void {}

fn main() {
    let x: Result<u32, &'static Void> = Ok(23);
    let _ = match x {   //~ ERROR non-exhaustive
        Ok(n) => n,
    };

    let x: &Void = unsafe { std::mem::uninitialized() };
    let _ = match x {}; //~ ERROR non-exhaustive

    let x: (Void,) = unsafe { std::mem::uninitialized() };
    let _ = match x {}; //~ ERROR non-exhaustive

    let x: [Void; 1] = unsafe { std::mem::uninitialized() };
    let _ = match x {}; //~ ERROR non-exhaustive

    let x: &[Void] = unsafe { std::mem::uninitialized() };
    let _ = match x {   //~ ERROR non-exhaustive
        &[] => (),
    };

    let x: Void = unsafe { std::mem::uninitialized() };
    let _ = match x {}; // okay

    let x: Result<u32, Void> = Ok(23);
    let _ = match x {   //~ ERROR non-exhaustive
        Ok(x) => x,
    };

    let x: Result<u32, Void> = Ok(23);
    let Ok(x) = x;
    //~^ ERROR refutable
}
