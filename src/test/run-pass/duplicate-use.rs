// xfail-test
extern mod std;

use list = std::map::chained;
use std::list;

fn main() {
    let _x: list::T<int, int> = list::mk();
}
