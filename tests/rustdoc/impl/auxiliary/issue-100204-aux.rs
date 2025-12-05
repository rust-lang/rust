#![crate_name="first"]

pub mod prelude {
    pub use crate::Bot;
}

pub struct Bot;

impl Bot {
    pub fn new() -> Bot {
        Bot
    }
}
