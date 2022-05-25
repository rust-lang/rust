use std::path::Path;

use ui_test::Comments;

#[test]
fn issue_2156() {
    let s = r"
use std::mem;

fn main() {
    let _x: &i32 = unsafe { mem::transmute(16usize) }; //~ ERROR encountered a dangling reference (address $HEX is unallocated)
}
    ";
    let comments = Comments::parse(Path::new("<dummy>"), s);
    println!("{:#?}", comments);
    assert_eq!(comments.error_matches[0].definition_line, 4);
    assert_eq!(comments.error_matches[0].revision, None);
    assert_eq!(
        comments.error_matches[0].matched,
        "encountered a dangling reference (address $HEX is unallocated)"
    );
}
