#![warn(clippy::iter_filter_is_some)]

fn main() {
    let _ = vec![Some(1)].into_iter().filter(Option::is_some);
    //~^ HELP: consider using `flatten` instead
    let _ = vec![Some(1)].into_iter().filter(|o| o.is_some());
    //~^ HELP: consider using `flatten` instead

    // Don't lint below
    let mut counter = 0;
    let _ = vec![Some(1)].into_iter().filter(|o| {
        counter += 1;
        o.is_some()
    });
    let _ = vec![Some(1)].into_iter().filter(|o| {
        // Roses are red,
        // Violets are blue,
        // `Err` is not an `Option`,
        // and this doesn't ryme
        o.is_some()
    });
}
