#![warn(clippy::result_filter_map)]
#![allow(clippy::map_flatten, clippy::unnecessary_map_on_constructor)]

fn odds_out(x: i32) -> Result<i32, ()> {
    if x % 2 == 0 { Ok(x) } else { Err(()) }
}

fn main() {
    // Unlike the Option version, Result does not have a filter operation => we will check for iterators
    // only
    let _ = vec![Ok(1) as Result<i32, ()>]
        .into_iter()
        .filter(Result::is_ok)
        //~^ result_filter_map
        .map(Result::unwrap);

    let _ = vec![Ok(1) as Result<i32, ()>]
        .into_iter()
        .filter(|o| o.is_ok())
        //~^ result_filter_map
        .map(|o| o.unwrap());

    let _ = vec![1]
        .into_iter()
        .map(odds_out)
        .filter(Result::is_ok)
        //~^ result_filter_map
        .map(Result::unwrap);
    let _ = vec![1]
        .into_iter()
        .map(odds_out)
        .filter(|o| o.is_ok())
        //~^ result_filter_map
        .map(|o| o.unwrap());
}
