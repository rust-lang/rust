#[macro_export]
macro_rules! emit_allow {
    () => {
        #[allow(unsafe_code)]
        let _so_safe = 0;
    };
}
