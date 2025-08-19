#![allow(
    clippy::let_underscore_untyped,
    clippy::missing_docs_in_private_items,
    clippy::map_identity,
    clippy::redundant_closure,
    clippy::unnecessary_wraps
)]

fn main() {
    // mapping to Option on Iterator
    fn option_id(x: i8) -> Option<i8> {
        Some(x)
    }
    let option_id_ref: fn(i8) -> Option<i8> = option_id;
    let option_id_closure = |x| Some(x);
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id).flatten().collect();
    //~^ map_flatten
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_ref).flatten().collect();
    //~^ map_flatten
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(option_id_closure).flatten().collect();
    //~^ map_flatten
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| x.checked_add(1)).flatten().collect();
    //~^ map_flatten

    // mapping to Iterator on Iterator
    let _: Vec<_> = vec![5_i8; 6].into_iter().map(|x| 0..x).flatten().collect();
    //~^ map_flatten

    // mapping to Option on Option
    let _: Option<_> = (Some(Some(1))).map(|x| x).flatten();
    //~^ map_flatten

    // mapping to Result on Result
    let _: Result<_, &str> = (Ok(Ok(1))).map(|x| x).flatten();
    //~^ map_flatten

    issue8734();
    issue8878();
}

fn issue8734() {
    let _ = [0u8, 1, 2, 3]
        .into_iter()
        .map(|n| match n {
            //~^ map_flatten
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
        //~^ map_flatten
// we need some newlines
// so that the span is big enough
// for a split output of the diagnostic
            Some("")
 // whitespace beforehand is important as well
        })
        .flatten();
}
