//@ run-fail
//@ compile-flags: -C debug-assertions -Zmir-opt-level=0 --emit=mir
//@ error-pattern: null pointer dereference

fn main() {
    // This is fine.
    // let cp: u32 = 12;
    // let _val: Bar = unsafe { std::mem::transmute::<u32, Bar>(cp) };

    // // This should break.
    let _val: Option<unsafe extern "C" fn(*mut u8)> =
        unsafe { std::mem::transmute::<u64, Option<unsafe extern "C" fn(*mut u8)>>(0) };
}
