#![warn(clippy::all, clippy::pedantic)]

fn main() {
    let a = ["1", "lol", "3", "NaN", "5"];

    let element: Option<i32> = a.iter().filter_map(|s| s.parse().ok()).next();
    assert_eq!(element, Some(1));

    #[rustfmt::skip]
    let _: Option<u32> = vec![1, 2, 3, 4, 5, 6]
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
