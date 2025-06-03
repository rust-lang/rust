//@ run-rustfix
//@ check-pass
#[allow(unreachable_code)]
#[allow(unused)]
#[warn(unused_braces)]
#[warn(unused_parens)]

fn main() {
    return { return }; //~ WARN: unnecessary braces
    if f({ return }) {} else {} //~ WARN: unnecessary braces
    if return { return } { return } else { return }
    match return { return } {
        _ => { return }
    }
    while return { return } {}
}

fn f(_a: ()) -> bool {
    true
}
