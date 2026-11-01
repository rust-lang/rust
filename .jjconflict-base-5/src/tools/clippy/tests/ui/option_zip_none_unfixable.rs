//@no-rustfix
#![warn(clippy::option_zip_none)]

fn main() {
    let _: Option<(i32, ())> = Option::zip(Some(1), None::<()>);
    //~^ option_zip_none
    let _: Option<((), i32)> = Option::zip(None::<()>, Some(1));
    //~^ option_zip_none
}
