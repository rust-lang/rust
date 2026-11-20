#![feature(min_specialization)]

trait Special {
    fn be_special();
}

impl<T> Special for T {
    fn be_special() {}
}

impl Special for usize {}
//~^ ERROR specialization impl does not specialize any associated items

fn main() {}
