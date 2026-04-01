trait HandlerFamily {
    type Target;
}

struct HandlerWrapper<H: HandlerFamily>(H);

impl<H: HandlerFamily> HandlerWrapper<H> {
    pub fn set_handler(&self, handler: &H::Target)
    where
        T: Send + Sync + 'static,
        //~^ ERROR cannot find type `T` in this scope
    {
    }
}

fn main() {}
