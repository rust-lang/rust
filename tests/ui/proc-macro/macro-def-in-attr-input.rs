//@ proc-macro: identity-attr.rs
//@ edition: 2021
//@ check-pass

// `gate_proc_macro_input` walks attribute-macro input with the
// `GateProcMacroInput` visitor; a `macro_rules!` definition inside the
// annotated item makes that walk hit `visit_macro_def`, which nothing
// else in the suite exercises.

#[identity_attr::id]
pub fn f() {
    macro_rules! m {
        () => {};
    }
    m!();
}

fn main() {}
