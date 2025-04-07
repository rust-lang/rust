#![allow(clippy::map_with_unused_argument_over_ranges)]
#![warn(clippy::suspicious_map)]

fn main() {
    let _ = (0..3).map(|x| x + 2).count();
    //~^ suspicious_map

    let f = |x| x + 1;
    let _ = (0..3).map(f).count();
    //~^ suspicious_map
}

fn negative() {
    // closure with side effects
    let mut sum = 0;
    let _ = (0..3).map(|x| sum += x).count();

    // closure variable with side effects
    let ext_closure = |x| sum += x;
    let _ = (0..3).map(ext_closure).count();

    // closure that returns unit
    let _ = (0..3)
        .map(|x| {
            // do nothing
        })
        .count();

    // external function
    let _ = (0..3).map(do_something).count();
}

fn do_something<T>(t: T) -> String {
    unimplemented!()
}
