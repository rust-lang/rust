#![warn(clippy::cmp_null)]

use std::ptr;

fn main() {
    let x = 0;
    let p: *const usize = &x;
    if p == ptr::null() {
        //~^ cmp_null

        println!("This is surprising!");
    }
    if ptr::null() == p {
        //~^ cmp_null

        println!("This is surprising!");
    }

    let mut y = 0;
    let m: *mut usize = &mut y;
    if m == ptr::null_mut() {
        //~^ cmp_null

        println!("This is surprising, too!");
    }
    if ptr::null_mut() == m {
        //~^ cmp_null

        println!("This is surprising, too!");
    }

    let _ = x as *const () == ptr::null();
    //~^ cmp_null
}

fn issue15010() {
    let f: *mut i32 = std::ptr::null_mut();
    debug_assert!(f != std::ptr::null_mut());
    //~^ cmp_null
}

fn issue16281() {
    use std::ptr;

    struct Container {
        value: *const i32,
    }
    let x = Container { value: ptr::null() };

    macro_rules! dot_value {
        ($obj:expr) => {
            $obj.value
        };
    }

    if dot_value!(x) == ptr::null() {
        //~^ cmp_null
        todo!()
    }
}
