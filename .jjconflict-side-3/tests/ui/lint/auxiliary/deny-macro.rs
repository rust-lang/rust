#[macro_export]
macro_rules! emit_deny {
    () => {
        #[deny(unsafe_code)]
        let _so_safe = 0;
    };
}
