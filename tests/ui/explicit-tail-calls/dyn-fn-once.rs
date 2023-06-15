#![feature(explicit_tail_calls)]

fn f() {
    become (Box::new(|| ()) as Box<dyn FnOnce() -> ()>)();
    //~^ error: mismatched function ABIs
    //~| error: mismatched signatures
}

fn g() {
    become (&g as &dyn FnOnce() -> ())();
    //~^ error: mismatched function ABIs
    //~| error: mismatched signatures
}

fn main() {}
