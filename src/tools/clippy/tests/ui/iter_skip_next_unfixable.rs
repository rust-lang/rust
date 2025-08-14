#![warn(clippy::iter_skip_next)]
#![allow(dead_code, clippy::iter_out_of_bounds)]
//@no-rustfix
/// Checks implementation of `ITER_SKIP_NEXT` lint
fn main() {
    // fix #8128
    let test_string = "1|1 2";
    let sp = test_string.split('|').map(|s| s.trim());
    let _: Vec<&str> = sp.skip(1).next().unwrap().split(' ').collect();
    //~^ iter_skip_next

    if let Some(s) = Some(test_string.split('|').map(|s| s.trim())) {
        let _: Vec<&str> = s.skip(1).next().unwrap().split(' ').collect();
        //~^ iter_skip_next
    };
    fn check<T>(s: T)
    where
        T: Iterator<Item = String>,
    {
        let _: Vec<&str> = s.skip(1).next().unwrap().split(' ').collect();
        //~^ iter_skip_next
    }
}
