// Test coercions between pointers which don't do anything fancy like unsizing.
// These are testing that we don't lose mutability when converting to raw pointers.

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
