//@ check-pass
//@aux-build:../auxiliary/proc_macros.rs
extern crate proc_macros;

proc_macros::with_span! {
    span
    #[cfg(test)]
    mod tests {}
}

#[test]
fn f() {}
