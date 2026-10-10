//@ edition:2021
//@ ignore-parallel-frontend query cycle
struct AnyOption<T>(T);
impl<T> AnyOption<T> {
    const NONE: Option<T> = None;
}

// This is an unfortunate side-effect of borrowchecking nested items
// together with their parent. Evaluating the `AnyOption::<_>::NONE`
// pattern for exhaustiveness checking relies on the layout of the
// async block. This layout relies on `optimized_mir` of the nested
// item which is now borrowck'd together with its parent. As
// borrowck of the parent requires us to have already lowered the match,
// this is a query cycle.
//
// Update: this no longer has cycle error after #164023

fn uwu() {}
fn defines() {
    match Some(async {}) {
        AnyOption::<_>::NONE => {}
        //~^ ERROR constant pattern cannot depend on generic parameters
        _ => {}
    }
}
fn main() {}
