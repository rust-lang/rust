struct NeedsDropTypes<'tcx, F>(std::marker::PhantomData<&'tcx F>);

impl<'tcx, F, I> Iterator for NeedsDropTypes<'tcx, F>
//~^ ERROR type annotations needed
where
    F: Fn(&Missing) -> Result<I, ()>,
    //~^ ERROR cannot find type `Missing` in this scope
    I: Iterator<Item = Missing>,
    //~^ ERROR cannot find type `Missing` in this scope
{}

fn main() {}
