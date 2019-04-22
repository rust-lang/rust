// rust-lang/rust#55492: errors detected during MIR-borrowck's
// analysis of a closure body may only be caught when AST-borrowck
// looks at some parent.

// revisions: migrate nll

// Since we are testing nll (and migration) explicitly as a separate
// revisions, don't worry about the --compare-mode=nll on this test.

// ignore-compare-mode-nll

//[nll]compile-flags: -Z borrowck=mir


// transcribed from borrowck-closures-unique.rs
mod borrowck_closures_unique {
    pub fn e(x: &'static mut isize) {
        static mut Y: isize = 3;
        let mut c1 = |y: &'static mut isize| x = y;
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
        unsafe { c1(&mut Y); }
    }
}

mod borrowck_closures_unique_grandparent {
    pub fn ee(x: &'static mut isize) {
        static mut Z: isize = 3;
        let mut c1 = |z: &'static mut isize| {
            let mut c2 = |y: &'static mut isize| x = y;
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
            c2(z);
        };
        unsafe { c1(&mut Z); }
    }
}

// adapted from mutability_errors.rs
mod mutability_errors {
    pub fn capture_assign_whole(x: (i32,)) {
        || { x = (1,); };
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
    }
    pub fn capture_assign_part(x: (i32,)) {
        || { x.0 = 1; };
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
    }
    pub fn capture_reborrow_whole(x: (i32,)) {
        || { &mut x; };
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
    }
    pub fn capture_reborrow_part(x: (i32,)) {
        || { &mut x.0; };
        //[migrate]~^ ERROR is not declared as mutable
        //[nll]~^^ ERROR is not declared as mutable
    }
}

fn main() {
    static mut X: isize = 2;
    unsafe { borrowck_closures_unique::e(&mut X); }

    mutability_errors::capture_assign_whole((1000,));
    mutability_errors::capture_assign_part((2000,));
    mutability_errors::capture_reborrow_whole((3000,));
    mutability_errors::capture_reborrow_part((4000,));
}
