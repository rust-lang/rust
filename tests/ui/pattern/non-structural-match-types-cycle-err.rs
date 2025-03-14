//@ edition:2021

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

fn uwu() {}
fn defines() {
    match Some(async {}) {
        AnyOption::<_>::NONE => {}
        //~^ ERROR cycle detected when building THIR for `defines`
        _ => {}
    }
}
fn main() {}
