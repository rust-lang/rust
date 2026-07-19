//@ edition:2021

// Test that we restrict precision of a capture when we access a raw ptr,
// i.e. the capture doesn't deref the raw ptr.


#![feature(rustc_attrs)]

#[derive(Debug)]
struct S {
    s: String,
    t: String,
}

struct T(*const S);

fn unsafe_imm() {
    let s = "".into();
    let t = "".into();
    let my_speed: Box<S> = Box::new(S { s, t });

    let p : *const S = Box::into_raw(my_speed);
    let t = T(p);

    let c = #[rustc_capture_analysis]
     || unsafe {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        println!("{:?}", (*t.0).s);
        //~^ NOTE: Capturing t[(0, 0),Deref,(0, 0)] -> Immutable
        //~| NOTE: Min Capture t[(0, 0)] -> Immutable
    };

    c();
}

fn unsafe_mut() {
    let s = "".into();
    let t = "".into();
    let mut my_speed: Box<S> = Box::new(S { s, t });
    let p : *mut S = &mut *my_speed;

    let c = #[rustc_capture_analysis]
    || {
    //~^ ERROR: First Pass analysis includes:
    //~| ERROR: Min Capture analysis includes:
        let x = unsafe { &mut (*p).s };
        //~^ NOTE: Capturing p[Deref,(0, 0)] -> Immutable
        //~| NOTE: Min Capture p[] -> Immutable
        *x = "s".into();
    };
    c();
}

fn main() {
    unsafe_mut();
    unsafe_imm();
}
