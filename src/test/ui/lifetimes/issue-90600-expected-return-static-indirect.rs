use std::cell::RefCell;
use std::io::Read;

fn main() {}

fn inner(mut foo: &[u8]) {
    let refcell = RefCell::new(&mut foo);
    //~^ ERROR `foo` does not live long enough
    let read = &refcell as &RefCell<dyn Read>;
    //~^ ERROR lifetime may not live long enough

    read_thing(read);
}

fn read_thing(refcell: &RefCell<dyn Read>) {}
