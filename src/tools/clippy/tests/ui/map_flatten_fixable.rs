// run-rustfix

#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::let_underscore_untyped)]
#![allow(clippy::missing_docs_in_private_items)]
#![allow(clippy::map_identity)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::unnecessary_wraps)]
#![feature(result_flattening)]

fn main() {
    // mapping to Option on Iterator
    fn option_id(x: i8) -> Option<i8> {
        Some(x)
    }
    let option_id_ref: fn(i8) -> Option<i8> = option_id;
    let option_id_closure = |x| Some(x);
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_ref).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_closure).flatten().collect();
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| x.checked_add(1)).flatten().collect();

    // mapping to Iterator on Iterator
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| 0..x).flatten().collect();

    // mapping to Option on Option
    let _: Option<_> = (Some(Some(1))).map(|x| x).flatten();

    // mapping to Result on Result
    let _: Result<_, &str> = (Ok(Ok(1))).map(|x| x).flatten();

    issue8734();
    issue8878();
}

fn issue8734() {
    let _ = [0u8, 1, 2, 3]
        .into_iter()
        .map(|n| match n {
            1 => [n
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)
                .saturating_add(1)],
            n => [n],
        })
        .flatten();
}

#[allow(clippy::bind_instead_of_map)] // map + flatten will be suggested to `and_then`, but afterwards `map` is suggested again
#[rustfmt::skip] // whitespace is important for this one
fn issue8878() {
    std::collections::HashMap::<u32, u32>::new()
        .get(&0)
        .map(|_| {
// we need some newlines
// so that the span is big enough
// for a split output of the diagnostic
            Some("")
 // whitespace beforehand is important as well
        })
        .flatten();
}
