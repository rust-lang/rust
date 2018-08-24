// #39872, #39553

fn will_ice(something: &u32) -> impl Iterator<Item = &u32> {
    //~^ ERROR the trait bound `(): std::iter::Iterator` is not satisfied [E0277]
}

fn main() {}
