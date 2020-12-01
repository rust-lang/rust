// run-pass
// compile-flags:-Zmir-opt-level=2

struct Cursor {}
struct TokenTree {}

impl Iterator for Cursor {
    type Item = TokenTree;

    fn next(&mut self) -> Option<TokenTree> {
        None
    }
}

fn tokenstream_probably_equal_for_proc_macro() {
    fn break_tokens(_tree: TokenTree) -> impl Iterator<Item = TokenTree> {
        let token_trees: Vec<TokenTree> = vec![];
        token_trees.into_iter()
    }

    let c1 = Cursor {};
    let c2 = Cursor {};

    let mut t1 = c1.flat_map(break_tokens);
    let mut t2 = c2.flat_map(break_tokens);

    for (_t1, _t2) in t1.by_ref().zip(t2.by_ref()) {}
}

fn main() {
    tokenstream_probably_equal_for_proc_macro();
}
