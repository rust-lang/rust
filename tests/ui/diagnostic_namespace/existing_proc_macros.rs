//@ check-pass
//@ proc-macro: proc-macro-helper.rs

extern crate proc_macro_helper;

mod test1 {
    use proc_macro_helper::diagnostic;

    #[diagnostic]
    struct Foo;

}

mod test2 {
    mod diagnostic {
        pub use proc_macro_helper::diagnostic as on_unimplemented;
    }

    #[diagnostic::on_unimplemented]
    trait Foo {}
}

fn main() {}
