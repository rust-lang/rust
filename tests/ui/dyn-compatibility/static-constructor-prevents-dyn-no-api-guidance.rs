trait Factory {
    fn create() -> Box<dyn Factory>;
    //~^ ERROR the trait `Factory` is not dyn compatible
}

fn use_factory(_: &dyn Factory) {}
//~^ ERROR the trait `Factory` is not dyn compatible

fn main() {}
