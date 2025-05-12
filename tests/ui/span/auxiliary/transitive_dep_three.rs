#[macro_export]
macro_rules! define_parse_error {
    () => {
        #[macro_export]
        macro_rules! parse_error {
            () => { parse error }
        }
    }
}
