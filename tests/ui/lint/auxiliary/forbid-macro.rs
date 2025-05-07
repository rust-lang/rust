#[macro_export]
macro_rules! emit_forbid {
    () => {
        #[forbid(unsafe_code)]
        let _so_safe = 0;
    };
}
