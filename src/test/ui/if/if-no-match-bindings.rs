// Checks for `if` expressions with respect to default match bindings.
// Specifically, we do not accept `if cond { ... }` where `cond: &mut? bool`.
// Meanwhile, `match cond { true => ..., _ => ... }` does accept that.

// FIXME(@rust-lang/lang-team): consider relaxing this?

fn b_ref<'a>() -> &'a bool { &true }
fn b_mut_ref<'a>() -> &'a mut bool { &mut true }

fn main() {
    // This is OK:
    match b_ref() { true => {}, _ => {} }
    match b_mut_ref() { true => {}, _ => {} }
    match &true { true => {}, _ => {} }
    match &mut true { true => {}, _ => {} }

    // This is NOT:
    if b_ref() {} //~ ERROR mismatched types [E0308]
    if b_mut_ref() {} //~ ERROR mismatched types [E0308]
    if &true {} //~ ERROR mismatched types [E0308]
    if &mut true {} //~ ERROR mismatched types [E0308]
}
