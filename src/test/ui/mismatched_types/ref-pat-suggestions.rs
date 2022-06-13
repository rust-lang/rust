// run-rustfix

fn _f0(&_a: u32) {} //~ ERROR mismatched types
fn _f1(&mut _a: u32) {} //~ ERROR mismatched types
fn _f2(&&_a: &u32) {} //~ ERROR mismatched types
fn _f3(&mut &_a: &mut u32) {} //~ ERROR mismatched types
fn _f4(&&mut _a: &u32) {} //~ ERROR mismatched types
fn _f5(&mut &mut _a: &mut u32) {} //~ ERROR mismatched types

fn main() {
    let _: fn(u32) = |&_a| (); //~ ERROR mismatched types
    let _: fn(u32) = |&mut _a| (); //~ ERROR mismatched types
    let _: fn(&u32) = |&&_a| (); //~ ERROR mismatched types
    let _: fn(&mut u32) = |&mut &_a| (); //~ ERROR mismatched types
    let _: fn(&u32) = |&&mut _a| (); //~ ERROR mismatched types
    let _: fn(&mut u32) = |&mut &mut _a| (); //~ ERROR mismatched types

    let _ = |&_a: u32| (); //~ ERROR mismatched types
    let _ = |&mut _a: u32| (); //~ ERROR mismatched types
    let _ = |&&_a: &u32| (); //~ ERROR mismatched types
    let _ = |&mut &_a: &mut u32| (); //~ ERROR mismatched types
    let _ = |&&mut _a: &u32| (); //~ ERROR mismatched types
    let _ = |&mut &mut _a: &mut u32| (); //~ ERROR mismatched types
}
