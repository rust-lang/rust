// run-rustfix
#![warn(clippy::option_filter_map)]
#![allow(clippy::map_flatten)]

fn main() {
    let _ = Some(Some(1)).filter(Option::is_some).map(Option::unwrap);
    let _ = Some(Some(1)).filter(|o| o.is_some()).map(|o| o.unwrap());
    let _ = Some(1).map(odds_out).filter(Option::is_some).map(Option::unwrap);
    let _ = Some(1).map(odds_out).filter(|o| o.is_some()).map(|o| o.unwrap());

    let _ = vec![Some(1)].into_iter().filter(Option::is_some).map(Option::unwrap);
    let _ = vec![Some(1)].into_iter().filter(|o| o.is_some()).map(|o| o.unwrap());
    let _ = vec![1]
        .into_iter()
        .map(odds_out)
        .filter(Option::is_some)
        .map(Option::unwrap);
    let _ = vec![1]
        .into_iter()
        .map(odds_out)
        .filter(|o| o.is_some())
        .map(|o| o.unwrap());
}

fn odds_out(x: i32) -> Option<i32> {
    if x % 2 == 0 { Some(x) } else { None }
}
