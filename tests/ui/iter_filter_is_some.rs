#![warn(clippy::iter_filter_is_some)]

fn odds_out(x: i32) -> Option<i32> {
    if x % 2 == 0 { Some(x) } else { None }
}

fn main() {
    let _ = vec![Some(1)].into_iter().filter(Option::is_some);
    //~^ HELP: consider using `flatten` instead
    let _ = vec![Some(1)].into_iter().filter(|o| o.is_some());
    //~^ HELP: consider using `flatten` instead
}
