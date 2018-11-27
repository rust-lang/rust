// edition:2018

// For the time being `macro_rules` items are treated as *very* private...

#![feature(decl_macro, uniform_paths)]

mod m1 {
    macro_rules! legacy_macro { () => () }

    // ... so they can't be imported by themselves, ...
    use legacy_macro as _; //~ ERROR `legacy_macro` is private, and cannot be re-exported
}

mod m2 {
    macro_rules! legacy_macro { () => () }

    type legacy_macro = u8;

    // ... but don't prevent names from other namespaces from being imported, ...
    use legacy_macro as _; // OK
}

mod m3 {
    macro legacy_macro() {}

    fn f() {
        macro_rules! legacy_macro { () => () }

        // ... but still create ambiguities with other names in the same namespace.
        use legacy_macro as _; //~ ERROR `legacy_macro` is ambiguous
                               //~| ERROR `legacy_macro` is private, and cannot be re-exported
    }
}

mod exported {
    // Exported macros are treated as private as well,
    // some better rules need to be figured out later.
    #[macro_export]
    macro_rules! legacy_macro { () => () }

    use legacy_macro as _; //~ ERROR `legacy_macro` is private, and cannot be re-exported
}

fn main() {}
