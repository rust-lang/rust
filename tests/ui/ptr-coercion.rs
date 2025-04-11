// Test coercions between pointers which don't do anything fancy like unsizing.
// These are testing that we don't lose mutability when converting to raw pointers.

pub fn main() {
    // *const -> *mut
    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| NOTE_NONVIRAL expected raw pointer `*mut isize`
                           //~| NOTE_NONVIRAL found raw pointer `*const isize`
                           //~| NOTE_NONVIRAL types differ in mutability

    // & -> *mut
    let x: *mut isize = &42; //~  ERROR mismatched types
                             //~| NOTE_NONVIRAL expected raw pointer `*mut isize`
                             //~| NOTE_NONVIRAL found reference `&isize`
                             //~| NOTE_NONVIRAL types differ in mutability

    let x: *const isize = &42;
    let x: *mut isize = x; //~  ERROR mismatched types
                           //~| NOTE_NONVIRAL expected raw pointer `*mut isize`
                           //~| NOTE_NONVIRAL found raw pointer `*const isize`
                           //~| NOTE_NONVIRAL types differ in mutability
}
