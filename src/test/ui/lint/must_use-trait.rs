#![deny(unused_must_use)]

#[must_use]
trait Critical {}

trait NotSoCritical {}

trait DecidedlyUnimportant {}

struct Anon;

impl Critical for Anon {}
impl NotSoCritical for Anon {}
impl DecidedlyUnimportant for Anon {}

fn get_critical() -> impl NotSoCritical + Critical + DecidedlyUnimportant {
    Anon {}
}

fn get_boxed_critical() -> Box<dyn Critical> {
    Box::new(Anon {})
}

fn get_nested_boxed_critical() -> Box<Box<dyn Critical>> {
    Box::new(Box::new(Anon {}))
}

fn get_critical_tuple() -> (u32, Box<dyn Critical>, impl Critical, ()) {
    (0, get_boxed_critical(), get_critical(), ())
}

fn main() {
    get_critical(); //~ ERROR unused implementer of `Critical` that must be used
    get_boxed_critical(); //~ ERROR unused boxed `Critical` trait object that must be used
    get_nested_boxed_critical();
    //~^ ERROR unused boxed boxed `Critical` trait object that must be used
    get_critical_tuple(); //~ ERROR unused boxed `Critical` trait object in tuple element 1
    //~^ ERROR unused implementer of `Critical` in tuple element 2
}
