//@ edition:2018

#![feature(decl_macro)]

mod m1 {
    // Non-exported legacy macros are treated as `pub(crate)`.
    macro_rules! legacy_macro { () => () }

    use legacy_macro as _; // OK
    pub(crate) use legacy_macro as _; // OK
    pub use legacy_macro as _; //~ ERROR `legacy_macro` is only public within the crate, and cannot be re-exported outside
}

mod m2 {
    macro_rules! legacy_macro { () => () }

    #[allow(non_camel_case_types)]
    type legacy_macro = u8;

    // Legacy macro imports don't prevent names from other namespaces from being imported.
    use legacy_macro as _; // OK
}

mod m3 {
    macro legacy_macro() {}

    fn f() {
        macro_rules! legacy_macro { () => () }

        use legacy_macro as _; // OK
    }
}

mod exported {
    // Exported legacy macros are treated as `pub`.
    #[macro_export]
    macro_rules! legacy_macro { () => () }

    pub use legacy_macro as _; // OK
}

fn main() {}
