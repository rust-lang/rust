/*!
 * Try to double-check that static fns have the right size (with or
 * without dummy env ptr, as appropriate) by iterating a size-2 array.
 * If the static size differs from the runtime size, the second element
 * should be read as a null or otherwise wrong pointer and crash.
 */

fn f() {}
static mut CLOSURES: &'static mut [fn()] = &mut [f as fn(), f as fn()];

pub fn main() {
    unsafe {
        for closure in &mut *CLOSURES {
            (*closure)()
        }
    }
}
