#![deny(empty_iterator_range)]

fn main() {
    for _i in 10..0 {
        //~^ ERROR this `for` loop is never run [empty_iterator_range]
    }
    for _i in (10..0).rev() {
        //~^ ERROR this `for` loop is never run [empty_iterator_range]
    }
    for _i in (10..0).step_by(1) {
        //~^ ERROR this `for` loop is never run [empty_iterator_range]
    }
}
