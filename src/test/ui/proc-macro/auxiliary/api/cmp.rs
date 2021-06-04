use proc_macro::{LineColumn, Punct, Spacing};

pub fn test() {
    test_line_column_ord();
    test_punct_eq();
}

fn test_line_column_ord() {
    let line0_column0 = LineColumn { line: 0, column: 0 };
    let line0_column1 = LineColumn { line: 0, column: 1 };
    let line1_column0 = LineColumn { line: 1, column: 0 };
    assert!(line0_column0 < line0_column1);
    assert!(line0_column1 < line1_column0);
}

fn test_punct_eq() {
    let colon_alone = Punct::new(':', Spacing::Alone);
    assert_eq!(colon_alone, ':');
    let colon_joint = Punct::new(':', Spacing::Joint);
    assert_eq!(colon_joint, ':');
}
