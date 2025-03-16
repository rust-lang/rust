#![warn(clippy::flat_map_option)]
#![allow(clippy::redundant_closure, clippy::unnecessary_filter_map)]

fn main() {
    // yay
    let c = |x| Some(x);
    let _ = [1].iter().flat_map(c);
    //~^ flat_map_option
    let _ = [1].iter().flat_map(Some);
    //~^ flat_map_option

    // nay
    let _ = [1].iter().flat_map(|_| &Some(1));
}
