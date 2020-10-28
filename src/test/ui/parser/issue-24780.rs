// Verify that '>' is not both expected and found at the same time, as it used
// to happen in #24780. For example, following should be an error:
// expected one of ..., `>`, ... found `>`. No longer exactly this, but keeping for posterity.

fn foo() -> Vec<usize>> { //~ ERROR unmatched angle bracket
    Vec::new()
}

fn main() {}
