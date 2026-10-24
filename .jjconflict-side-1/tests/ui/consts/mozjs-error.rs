//@ run-pass
#![allow(dead_code)]
#![allow(non_upper_case_globals)]

struct CustomAutoRooterVFTable {
    trace: unsafe extern "C" fn(this: *mut i32, trc: *mut u32),
}

unsafe trait CustomAutoTraceable: Sized {
    const vftable: CustomAutoRooterVFTable = CustomAutoRooterVFTable {
        trace: Self::trace,
    };

    unsafe extern "C" fn trace(this: *mut i32, trc: *mut u32) {
        let this = this as *const Self;
        let this = this.as_ref().unwrap();
        Self::do_trace(this, trc);
    }

    fn do_trace(&self, trc: *mut u32);
}

unsafe impl CustomAutoTraceable for () {
    fn do_trace(&self, _: *mut u32) {
        // nop
    }
}

fn main() {
    let _ = <()>::vftable;
}
