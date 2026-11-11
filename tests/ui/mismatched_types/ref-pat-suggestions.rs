//@ run-rustfix

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

    #[allow(unused_mut)]
    {
        struct S(u8);

        let &mut _a = 0; //~ ERROR mismatched types
        let S(&mut _b) = S(0); //~ ERROR mismatched types
        let (&mut _c,) = (0,); //~ ERROR mismatched types

        match 0 {
            &mut _d => {} //~ ERROR mismatched types
        }
    }
}
