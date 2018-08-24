// Test coercions between pointers which don't do anything fancy like unsizing.
// These are testing that we don't lose mutability when converting to raw pointers.

pub fn main() {
    // *const -> *mut
    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| expected type `*mut isize`
                           //~| found type `*const isize`
                           //~| types differ in mutability

    // & -> *mut
    let x: *mut isize = &42; //~  ERROR mismatched types
                             //~| expected type `*mut isize`
                             //~| found type `&isize`
                             //~| types differ in mutability

    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| expected type `*mut isize`
                           //~| found type `*const isize`
                           //~| types differ in mutability
}
