pub enum SomeEnum {
    B = SomeEnum::A, //~ ERROR no variant or associated item named `A` found for type `SomeEnum`
}

fn main() {}
