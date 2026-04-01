use std::fmt::{Display, Error, Formatter};

// This test case exercises std::sys_common::remutex::ReentrantMutex
// by calling println!() from inside fmt.

struct InterruptingCow;

impl Display for InterruptingCow {
    fn fmt(&self, _f: &mut Formatter<'_>) -> Result<(), Error> {
        println!("Moo");
        Ok(())
    }
}

fn main() {
    println!("\"Knock knock\" \"Who's {} there?\"", InterruptingCow);
}
