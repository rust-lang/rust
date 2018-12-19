pub trait Trait {
    fn dyn_instead_of_self(self: Box<dyn Trait>);
    //~^ ERROR invalid method receiver type: std::boxed::Box<(dyn Trait + 'static)>
}

pub fn main() {
}
