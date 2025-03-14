#![allow(clippy::bind_instead_of_map)]

fn main() {
    let opt = Some(1);
    let r: Result<i32, i32> = Ok(1);
    let bar = |_| Some(1);

    // Check `OPTION_MAP_OR_NONE`.
    // Single line case.
    let _: Option<i32> = opt.map_or(None, |x| Some(x + 1));
    //~^ option_map_or_none
    // Multi-line case.
    #[rustfmt::skip]
    let _: Option<i32> = opt.map_or(None, |x| {
    //~^ option_map_or_none
                        Some(x + 1)
                       });
    // function returning `Option`
    let _: Option<i32> = opt.map_or(None, bar);
    //~^ option_map_or_none
    let _: Option<i32> = opt.map_or(None, |x| {
        //~^ option_map_or_none
        let offset = 0;
        let height = x;
        Some(offset + height)
    });

    // Check `RESULT_MAP_OR_INTO_OPTION`.
    let _: Option<i32> = r.map_or(None, Some);
    //~^ result_map_or_into_option
}
