use std::cell::UnsafeCell;

fn main() {
    unsafe {
        let c = &UnsafeCell::new(UnsafeCell::new(0));
        let inner_uniq = &mut *c.get();
        // stack: [c: SharedReadWrite, inner_uniq: Unique]

        let inner_shr = &*inner_uniq; // adds a SharedReadWrite
        // stack: [c: SharedReadWrite, inner_uniq: Unique, inner_shr: SharedReadWrite]

        *c.get() = UnsafeCell::new(1); // invalidates inner_shr
        // stack: [c: SharedReadWrite]

        let _val = *inner_shr.get(); //~ ERROR: /retag .* tag does not exist in the borrow stack/
    }
}
