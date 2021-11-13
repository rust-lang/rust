use std::cell::RefCell;
use std::io::Read;

fn main() {}

fn inner(mut foo: &[u8]) {
    let refcell = RefCell::new(&mut foo);
    let read = &refcell as &RefCell<dyn Read>;

    read_thing(read);
    //~^ ERROR explicit lifetime required in the type of `foo` [E0621]
}

fn read_thing(refcell: &RefCell<dyn Read>) {}
