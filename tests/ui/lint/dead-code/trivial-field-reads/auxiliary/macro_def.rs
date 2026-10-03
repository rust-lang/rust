#[macro_export]
macro_rules! bar {
    ($name: ident) => {
        #[rustc_trivial_field_reads]
        fn bar(s: &$name) {
            println!("{}", s.0);
        }
    };
}
