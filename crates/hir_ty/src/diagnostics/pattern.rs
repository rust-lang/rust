#![deny(elided_lifetimes_in_paths)]
#![allow(unused)] // todo remove

mod deconstruct_pat;
pub mod usefulness;

#[cfg(test)]
mod tests {
    use crate::diagnostics::tests::check_diagnostics;

    use super::*;

    #[test]
    fn unit_exhaustive() {
        check_diagnostics(
            r#"
fn main() {
    match ()   { ()   => {} }
    match ()   { _    => {} }
}
"#,
        );
    }

    #[test]
    fn unit_non_exhaustive() {
        check_diagnostics(
            r#"
fn main() {
    match ()   {            }
        //^^ Missing match arm
}
"#,
        );
    }
}
