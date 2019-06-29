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

fn main() {
    get_critical(); //~ ERROR unused implementer of `Critical` that must be used
    get_boxed_critical(); //~ ERROR unused boxed `Critical` trait object that must be used
}
