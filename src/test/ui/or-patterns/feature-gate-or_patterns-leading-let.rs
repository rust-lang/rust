// Test feature gating for a sole leading `|` in `let`.

fn main() {}

#[cfg(FALSE)]
fn gated_leading_vert_in_let() {
    let | A; //~ ERROR or-patterns syntax is experimental
    //~^ ERROR top-level or-patterns are not allowed
}
