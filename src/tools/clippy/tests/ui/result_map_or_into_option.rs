#![warn(clippy::result_map_or_into_option)]

fn main() {
    let opt: Result<u32, &str> = Ok(1);
    let _ = opt.map_or(None, Some);
    //~^ ERROR: called `map_or(None, Some)` on a `Result` value
    let _ = opt.map_or_else(|_| None, Some);
    //~^ ERROR: called `map_or_else(|_| None, Some)` on a `Result` value
    #[rustfmt::skip]
    let _ = opt.map_or_else(|_| { None }, Some);
    //~^ ERROR: called `map_or_else(|_| None, Some)` on a `Result` value

    let rewrap = |s: u32| -> Option<u32> { Some(s) };

    // A non-Some `f` arg should not emit the lint
    let opt: Result<u32, &str> = Ok(1);
    let _ = opt.map_or(None, rewrap);

    // A non-Some `f` closure where the argument is not used as the
    // return should not emit the lint
    let opt: Result<u32, &str> = Ok(1);
    _ = opt.map_or(None, |_x| Some(1));
    let _ = opt.map_or_else(|a| a.parse::<u32>().ok(), Some);
}
