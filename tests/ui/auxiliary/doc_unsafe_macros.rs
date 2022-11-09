#[macro_export]
macro_rules! undocd_unsafe {
    () => {
        pub unsafe fn oy_vey() {
            unimplemented!();
        }
    };
}
#[macro_export]
macro_rules! undocd_safe {
    () => {
        pub fn vey_oy() {
            unimplemented!();
        }
    };
}
