//! Tests that pointer coercions preserving mutability are enforced:

//@ dont-require-annotations: NOTE

pub fn main() {
    // *const -> *mut
    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| NOTE expected raw pointer `*mut isize`
                           //~| NOTE found raw pointer `*const isize`
                           //~| NOTE types differ in mutability

    // & -> *mut
    let x: *mut isize = &42; //~  ERROR mismatched types
                             //~| NOTE expected raw pointer `*mut isize`
                             //~| NOTE found reference `&isize`
                             //~| NOTE types differ in mutability

    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| NOTE expected raw pointer `*mut isize`
                           //~| NOTE found raw pointer `*const isize`
                           //~| NOTE types differ in mutability
}
