#![warn(clippy::iter_filter_is_ok)]

fn main() {
    let _ = vec![Ok(1), Err(2), Ok(3)].into_iter().filter(Result::is_ok);
    //~^ HELP: consider using `flatten` instead
    let _ = vec![Ok(1), Err(2), Ok(3)].into_iter().filter(|a| a.is_ok());
    //~^ HELP: consider using `flatten` instead

    #[rustfmt::skip]
    let _ = vec![Ok(1), Err(2)].into_iter().filter(|o| { o.is_ok() });
    //~^ HELP: consider using `flatten` instead

    // Don't lint below
    let mut counter = 0;
    let _ = vec![Ok(1), Err(2)].into_iter().filter(|o| {
        counter += 1;
        o.is_ok()
    });
    let _ = vec![Ok(1), Err(2)].into_iter().filter(|o| {
        // Roses are red,
        // Violets are blue,
        // `Err` is not an `Option`,
        // and this doesn't ryme
        o.is_ok()
    });
}
