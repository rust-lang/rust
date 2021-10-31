// compile-flags: --crate-type=lib
#![feature(const_mut_refs)]
#![feature(const_precise_live_drops)]
#![feature(const_swap)]

// Mutable borrow of a field with drop impl.
pub const fn f() {
    let mut a: (u32, Option<String>) = (0, None); //~ ERROR destructors cannot be evaluated
    let _ = &mut a.1;
}

// Mutable borrow of a type with drop impl.
pub const A1: () = {
    let mut x = None; //~ ERROR destructors cannot be evaluated
    let mut y = Some(String::new());
    let a = &mut x;
    let b = &mut y;
    std::mem::swap(a, b);
    std::mem::forget(y);
};

// Mutable borrow of a type with drop impl.
pub const A2: () = {
    let mut x = None;
    let mut y = Some(String::new());
    let a = &mut x;
    let b = &mut y;
    std::mem::swap(a, b);
    std::mem::forget(y);
    let _z = x; //~ ERROR destructors cannot be evaluated
};

// Shared borrow of a type that might be !Freeze and Drop.
pub const fn g1<T>() {
    let x: Option<T> = None; //~ ERROR destructors cannot be evaluated
    let _ = x.is_some();
}

// Shared borrow of a type that might be !Freeze and Drop.
pub const fn g2<T>() {
    let x: Option<T> = None;
    let _ = x.is_some();
    let _y = x; //~ ERROR destructors cannot be evaluated
}
