#![feature(pattern_parentheses)]

fn main() {
    match 0 {
        (pat) => assert_eq!(pat, 0)
    }
}
