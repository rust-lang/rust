#![allow(clippy::mut_mut)]

fn takes_ref(a: &i32) {}
fn takes_refmut(a: &mut i32) {}
fn takes_ref_ref(a: &&i32) {}
fn takes_refmut_ref(a: &mut &i32) {}
fn takes_ref_refmut(a: &&mut i32) {}
fn takes_refmut_refmut(a: &mut &mut i32) {}
fn takes_raw_const(a: *const i32) {}
fn takes_raw_mut(a: *mut i32) {}

mod issue11268 {
    macro_rules! x {
        (1 $f:expr) => {
            $f(&mut 1);
        };
        (2 $f:expr) => {
            $f(&mut &1)
        };
        (3 $f:expr) => {
            $f(&mut &mut 1)
        };
        (4 $f:expr) => {
            let mut a = 1;
            $f(&raw mut a)
        };
    }

    fn f() {
        x!(1 super::takes_ref);
        x!(1 super::takes_refmut);
        x!(2 super::takes_refmut_ref);
        x!(3 super::takes_ref_refmut);
        x!(3 super::takes_refmut_refmut);
        x!(4 super::takes_raw_const);
        x!(4 super::takes_raw_mut);
    }
}

struct MyStruct;

impl MyStruct {
    fn takes_nothing(&self) {}
    fn takes_ref(&self, a: &i32) {}
    fn takes_refmut(&self, a: &mut i32) {}
    fn takes_ref_ref(&self, a: &&i32) {}
    fn takes_refmut_ref(&self, a: &mut &i32) {}
    fn takes_ref_refmut(&self, a: &&mut i32) {}
    fn takes_refmut_refmut(&self, a: &mut &mut i32) {}
    fn takes_raw_const(&self, a: *const i32) {}
    fn takes_raw_mut(&self, a: *mut i32) {}
}

#[warn(clippy::unnecessary_mut_passed)]
fn main() {
    // Functions
    takes_ref(&mut 42);
    //~^ unnecessary_mut_passed
    takes_ref_ref(&mut &42);
    //~^ unnecessary_mut_passed
    takes_ref_refmut(&mut &mut 42);
    //~^ unnecessary_mut_passed
    takes_raw_const(&mut 42);
    //~^ unnecessary_mut_passed

    let as_ptr: fn(&i32) = takes_ref;
    as_ptr(&mut 42);
    //~^ unnecessary_mut_passed
    let as_ptr: fn(&&i32) = takes_ref_ref;
    as_ptr(&mut &42);
    //~^ unnecessary_mut_passed
    let as_ptr: fn(&&mut i32) = takes_ref_refmut;
    as_ptr(&mut &mut 42);
    //~^ unnecessary_mut_passed
    let as_ptr: fn(*const i32) = takes_raw_const;
    as_ptr(&mut 42);
    //~^ unnecessary_mut_passed

    // Methods
    let my_struct = MyStruct;
    my_struct.takes_ref(&mut 42);
    //~^ unnecessary_mut_passed
    my_struct.takes_ref_ref(&mut &42);
    //~^ unnecessary_mut_passed
    my_struct.takes_ref_refmut(&mut &mut 42);
    //~^ unnecessary_mut_passed
    my_struct.takes_raw_const(&mut 42);
    //~^ unnecessary_mut_passed

    // No error

    // Functions
    takes_ref(&42);
    let as_ptr: fn(&i32) = takes_ref;
    as_ptr(&42);

    takes_refmut(&mut 42);
    let as_ptr: fn(&mut i32) = takes_refmut;
    as_ptr(&mut 42);

    takes_ref_ref(&&42);
    let as_ptr: fn(&&i32) = takes_ref_ref;
    as_ptr(&&42);

    takes_refmut_ref(&mut &42);
    let as_ptr: fn(&mut &i32) = takes_refmut_ref;
    as_ptr(&mut &42);

    takes_ref_refmut(&&mut 42);
    let as_ptr: fn(&&mut i32) = takes_ref_refmut;
    as_ptr(&&mut 42);

    takes_refmut_refmut(&mut &mut 42);
    let as_ptr: fn(&mut &mut i32) = takes_refmut_refmut;
    as_ptr(&mut &mut 42);

    takes_raw_const(&42);
    let as_ptr: fn(*const i32) = takes_raw_const;
    as_ptr(&42);

    takes_raw_mut(&mut 42);
    let as_ptr: fn(*mut i32) = takes_raw_mut;
    as_ptr(&mut 42);

    let a = &mut 42;
    let b = &mut &42;
    let c = &mut &mut 42;
    takes_ref(a);
    takes_ref_ref(b);
    takes_ref_refmut(c);
    takes_raw_const(a);

    // Methods
    my_struct.takes_ref(&42);
    my_struct.takes_refmut(&mut 42);
    my_struct.takes_ref_ref(&&42);
    my_struct.takes_refmut_ref(&mut &42);
    my_struct.takes_ref_refmut(&&mut 42);
    my_struct.takes_refmut_refmut(&mut &mut 42);
    my_struct.takes_raw_const(&42);
    my_struct.takes_raw_mut(&mut 42);
    my_struct.takes_ref(a);
    my_struct.takes_ref_ref(b);
    my_struct.takes_ref_refmut(c);
    my_struct.takes_raw_const(a);
    my_struct.takes_raw_mut(a);
}

// not supported currently
fn raw_ptrs(my_struct: MyStruct) {
    let mut n = 42;

    takes_raw_const(&raw mut n);

    let as_ptr: fn(*const i32) = takes_raw_const;
    as_ptr(&raw mut n);

    my_struct.takes_raw_const(&raw mut n);

    // No error

    takes_raw_const(&raw const n);
    takes_raw_mut(&raw mut n);

    let a = &raw mut n;
    takes_raw_const(a);

    my_struct.takes_raw_const(&raw const n);
    my_struct.takes_raw_mut(&raw mut n);
    my_struct.takes_raw_const(a);
}

#[expect(clippy::needless_borrow)]
fn issue15722(mut my_struct: MyStruct) {
    (&mut my_struct).takes_nothing();
    //~^ unnecessary_mut_passed
    (&my_struct).takes_nothing();

    // Only put parens around the argument that used to have it
    (&mut my_struct).takes_ref(&mut 42);
    //~^ unnecessary_mut_passed
    //~| unnecessary_mut_passed
    #[expect(clippy::double_parens)]
    (&mut my_struct).takes_ref((&mut 42));
    //~^ unnecessary_mut_passed
    //~| unnecessary_mut_passed
    #[expect(clippy::double_parens)]
    my_struct.takes_ref((&mut 42));
    //~^ unnecessary_mut_passed
}
