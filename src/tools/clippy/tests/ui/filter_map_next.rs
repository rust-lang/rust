#![warn(clippy::filter_map_next)]

fn main() {
    let a = ["1", "lol", "3", "NaN", "5"];

    #[rustfmt::skip]
    let _: Option<u32> = vec![1, 2, 3, 4, 5, 6]
    //~^ filter_map_next


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
