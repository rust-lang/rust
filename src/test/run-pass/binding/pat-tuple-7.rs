// run-pass

fn main() {
    #[allow(unused_parens)]
    match 0 {
        (pat) => assert_eq!(pat, 0)
    }
}
