// Verify that '>' is not both expected and found at the same time, as it used
// to happen in #24780. For example, following should be an error:
// expected one of ..., `>`, ... found `>`.

fn foo() -> Vec<usize>> { //~ ERROR expected one of `!`, `+`, `::`, `;`, `where`, or `{`, found `>`
    Vec::new()
}

fn main() {}
