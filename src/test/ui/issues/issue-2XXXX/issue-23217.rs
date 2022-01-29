pub enum SomeEnum {
    B = SomeEnum::A, //~ ERROR no variant or associated item named `A` found
}

fn main() {}
