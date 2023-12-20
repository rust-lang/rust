#![warn(clippy::iter_filter_is_ok)]

fn main() {
    let _ = vec![Ok(1), Err(2), Ok(3)].into_iter().filter(Result::is_ok);
    //~^ HELP: consider using `flatten` instead
    let _ = vec![Ok(1), Err(2), Ok(3)].into_iter().filter(|a| a.is_ok());
    //~^ HELP: consider using `flatten` instead
}
