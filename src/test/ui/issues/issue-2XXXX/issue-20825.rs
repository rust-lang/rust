pub trait Subscriber {
    type Input;
}

pub trait Processor: Subscriber<Input = Self::Input> {
    //~^ ERROR cycle detected
    type Input;
}

fn main() {}
