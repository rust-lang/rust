use std::cell::RefCell;
use std::io::Read;

fn main() {}

fn inner(mut foo: &[u8]) {
    let refcell = RefCell::new(&mut foo);
    //~^ ERROR `foo` has an anonymous lifetime `'_` but it needs to satisfy a `'static` lifetime requirement [E0759]
    let read = &refcell as &RefCell<dyn Read>;

    read_thing(read);
}

fn read_thing(refcell: &RefCell<dyn Read>) {}
