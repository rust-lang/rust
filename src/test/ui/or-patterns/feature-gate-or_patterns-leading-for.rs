// Test feature gating for a sole leading `|` in `let`.

fn main() {}

#[cfg(FALSE)]
fn gated_leading_vert_in_let() {
    for | A in 0 {} //~ ERROR or-patterns syntax is experimental
}
