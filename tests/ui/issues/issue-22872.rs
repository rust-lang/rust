trait Wrap<'b> {
    fn foo(&'b mut self);
}

struct Wrapper<P>(P);

impl<'b, P> Wrap<'b> for Wrapper<P>
where P: Process<'b>,
      <P as Process<'b>>::Item: Iterator {
    fn foo(&mut self) {}
}


pub trait Process<'a> {
    type Item;
    fn bar(&'a self);
}

fn push_process<P>(process: P) where P: Process<'static> {
    let _: Box<dyn for<'b> Wrap<'b>> = Box::new(Wrapper(process));
//~^ ERROR is not an iterator
}

fn main() {}
