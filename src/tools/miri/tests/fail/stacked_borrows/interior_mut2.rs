use std::cell::UnsafeCell;
use std::mem;

// Like `&mut *x.get()`, but without intermediate raw pointers.
#[allow(mutable_transmutes)]
unsafe fn unsafe_cell_get<T>(x: &UnsafeCell<T>) -> &'static mut T {
    mem::transmute(x)
}

fn main() {
    unsafe {
        let c = &UnsafeCell::new(UnsafeCell::new(0));
        let inner_uniq = &mut *c.get();
        let inner_shr = &*inner_uniq;
        // stack: [c: SharedReadWrite, inner_uniq: Unique, inner_shr: SharedReadWrite]

        let _val = c.get().read(); // invalidates inner_uniq
        // stack: [c: SharedReadWrite]

        // We have to be careful not to add any raw pointers above inner_uniq in
        // the stack, hence the use of unsafe_cell_get.
        // This used to work, but since we removed the "quirk" it fails here.
        let _val = *unsafe_cell_get(inner_shr); //~ ERROR: /retag .* tag does not exist in the borrow stack/

        *c.get() = UnsafeCell::new(0); // now inner_shr gets invalidated
        // stack: [c: SharedReadWrite]

        // this definitely should not work
        let _val = *inner_shr.get();
    }
}
