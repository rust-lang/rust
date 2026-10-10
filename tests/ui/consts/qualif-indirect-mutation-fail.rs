//@ compile-flags: --crate-type=lib
#![feature(const_precise_live_drops)]

struct NotConstDestruct;

impl Drop for NotConstDestruct {
    fn drop(&mut self) {}
}

// Mutable borrow of a field with drop impl.
pub const fn f() {
    let mut a: (u32, Option<NotConstDestruct>) = (0, None); //~ ERROR destructor of
    let _ = &mut a.1;
}

// FIXME: we make these associated consts to work around
// <https://github.com/rust-lang/rust/issues/163973>.
impl NotConstDestruct {
    // Mutable borrow of a type with drop impl.
    pub const A1: () = {
        let mut x = None; //~ ERROR destructor of
        let mut y = Some(NotConstDestruct);
        let a = &mut x;
        let b = &mut y;
        std::mem::swap(a, b);
        std::mem::forget(y);
    };

    // Mutable borrow of a type with drop impl.
    pub const A2: () = {
        let mut x = None;
        let mut y = Some(NotConstDestruct);
        let a = &mut x;
        let b = &mut y;
        std::mem::swap(a, b);
        std::mem::forget(y);
        let _z = x; //~ ERROR destructor of
    };
}

// Shared borrow of a type that might be !Freeze and Drop.
pub const fn g1<T>() {
    let x: Option<T> = None; //~ ERROR destructor of
    let _ = x.is_some();
}

// Shared borrow of a type that might be !Freeze and Drop.
pub const fn g2<T>() {
    let x: Option<T> = None;
    let _ = x.is_some();
    let _y = x; //~ ERROR destructor of
}

// Mutable raw reference to a Drop type.
pub const fn address_of_mut() {
    let mut x: Option<NotConstDestruct> = None; //~ ERROR destructor of
    &raw mut x;

    let mut y: Option<NotConstDestruct> = None; //~ ERROR destructor of
    std::ptr::addr_of_mut!(y);
}

// Const raw reference to a Drop type. Conservatively assumed to allow mutation
// until resolution of https://github.com/rust-lang/rust/issues/56604.
pub const fn address_of_const() {
    let x: Option<NotConstDestruct> = None; //~ ERROR destructor of
    &raw const x;

    let y: Option<NotConstDestruct> = None; //~ ERROR destructor of
    std::ptr::addr_of!(y);
}

// Regression test for an incorrect implementation of state join.
pub const fn regression_test_for_incorrect_join() {
    let mut a = Some(NotConstDestruct); //~ ERROR destructor of
    let b = NotConstDestruct;
    let p = &raw mut a;
    let x = unsafe { std::ptr::read(p) };
    std::mem::forget(a);
    std::mem::forget(x);
    a = None;
    unsafe { std::ptr::write(p, Some(b)) };
}

// Use after move cannot sneak past checks.
// NB: Semantics of MIR move elimination might make it possible to accept this.
pub const fn use_after_move() {
    let mut a = None; //~ ERROR destructor of
    let p = &raw mut a;
    std::mem::forget(a);
    a = None;
    unsafe { std::ptr::write(p, Some(NotConstDestruct)) };
}
