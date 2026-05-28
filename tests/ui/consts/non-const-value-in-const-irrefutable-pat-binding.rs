// Irrefutable `if let` pattern bindings still produce an incorrect suggestion
// to replace `let` with `const`.
// See https://github.com/rust-lang/rust/pull/152834#discussion_r3068766148

//@ known-bug: #152831

fn irrefutable_if_let_binding() {
    if let x = 1 {
        const { x }
    }
}

fn main() {}
