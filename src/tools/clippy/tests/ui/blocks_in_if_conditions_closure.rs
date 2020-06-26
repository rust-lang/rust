#![warn(clippy::blocks_in_if_conditions)]
#![allow(unused, clippy::let_and_return)]

fn predicate<F: FnOnce(T) -> bool, T>(pfn: F, val: T) -> bool {
    pfn(val)
}

fn pred_test() {
    let v = 3;
    let sky = "blue";
    // This is a sneaky case, where the block isn't directly in the condition,
    // but is actually inside a closure that the condition is using.
    // The same principle applies -- add some extra expressions to make sure
    // linter isn't confused by them.
    if v == 3
        && sky == "blue"
        && predicate(
            |x| {
                let target = 3;
                x == target
            },
            v,
        )
    {}

    if predicate(
        |x| {
            let target = 3;
            x == target
        },
        v,
    ) {}
}

fn closure_without_block() {
    if predicate(|x| x == 3, 6) {}
}

fn macro_in_closure() {
    let option = Some(true);

    if option.unwrap_or_else(|| unimplemented!()) {
        unimplemented!()
    }
}

fn main() {}
