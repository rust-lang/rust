#![feature(rustc_attrs)]

#[macro_export]
macro_rules! make_fn {
    ($name: ident) => {
        #[rustc_trivial_field_reads]
        pub fn foo() {
            println!("{}", $name(0).0);
        }
    }
}
