//@ run-rustfix
//@ check-pass
#[allow(unreachable_code)]
#[warn(unused_braces)]

fn main() {
    return { return }; //~ WARN: unnecessary braces
    if return { return } { return } else { return }
    match return { return } {
        _ => { return }
    }
    while return { return } {}
}
