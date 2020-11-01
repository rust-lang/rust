#![feature(proc_macro_span)]

use proc_macro::{Ident, LineColumn};

#[test]
fn test_line_column_ord() {
    let line0_column0 = LineColumn { line: 0, column: 0 };
    let line0_column1 = LineColumn { line: 0, column: 1 };
    let line1_column0 = LineColumn { line: 1, column: 0 };
    assert!(line0_column0 < line0_column1);
    assert!(line0_column1 < line1_column0);
}

#[test]
fn test_ident_eq() {
    // Good enough if it typechecks, since proc_macro::Ident can't exist in a test.
    fn _check(ident: &Ident, string: String) {
        let _ = ident == "serde";
        let _ = *ident == "serde";
        let _ = *ident == string;
        let _ = *ident == &&&string;
    }
}
