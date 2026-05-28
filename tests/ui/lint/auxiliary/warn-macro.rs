#[macro_export]
macro_rules! emit_warn {
    () => {
        #[warn(unsafe_code)]
        let _so_safe = 0;
    };
}
