#![feature(unsafe_binders)]

const None: Option<unsafe<> Option<Box<dyn Send>>> = None;
//~^ ERROR the trait bound `Box<(dyn Send + 'static)>: Copy` is not satisfied
//~| ERROR the trait bound `Box<(dyn Send + 'static)>: Copy` is not satisfied

fn main() {
    match None {
        _ => {}
    }
}
