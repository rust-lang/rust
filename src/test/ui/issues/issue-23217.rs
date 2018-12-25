pub enum SomeEnum {
    B = SomeEnum::A,
    //~^ ERROR no variant named `A` found for type `SomeEnum`
}

fn main() {}
