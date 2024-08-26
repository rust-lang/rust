struct Hello<'a> {
    value: Box<dyn std::any::Any + 'a>,
    //~^ ERROR lifetime bound not satisfied
}

impl<'a> Hello<'a> {
    fn new<T: 'a>(value: T) -> Self {
        Self { value: Box::new(value) }
        //~^ ERROR the parameter type `T` may not live long enough
    }
}

fn main() {}
