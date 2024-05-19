#![warn(clippy::all, clippy::pedantic)]

fn main() {
    let a = ["1", "lol", "3", "NaN", "5"];

    #[rustfmt::skip]
    let _: Option<u32> = vec![1, 2, 3, 4, 5, 6]
    //~^ ERROR: called `filter_map(..).next()` on an `Iterator`. This is more succinctly e
    //~| NOTE: `-D clippy::filter-map-next` implied by `-D warnings`
        .into_iter()
        .filter_map(|x| {
            if x == 2 {
                Some(x * 2)
            } else {
                None
            }
        })
        .next();
}
