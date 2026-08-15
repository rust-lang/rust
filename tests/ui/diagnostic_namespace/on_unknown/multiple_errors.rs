#![feature(diagnostic_on_unknown)]

mod test1 {
    #[diagnostic::on_unknown(
        message = "custom message",
        label = "custom label",
        note = "custom note"
    )]
    use std::vec::{NonExisting, Vec, Whatever};
    //~^ ERROR: custom message
}

mod test2 {
    #[diagnostic::on_unknown(
        message = "custom message",
        label = "custom label",
        note = "custom note"
    )]
    use std::{Whatever, vec::NonExisting, vec::Vec, *};
    //~^ ERROR: custom message
}

mod test3 {
    #[diagnostic::on_unknown(
        message = "custom message",
        label = "custom label",
        note = "custom note"
    )]
    use std::{
        string::String,
        vec::{NonExisting, Vec},
        //~^ ERROR: custom message
    };
}

mod test4 {
    #[diagnostic::on_unknown(
        message = "custom message",
        label = "custom label",
        note = "custom note"
    )]
    use std::{
        string::String,
        vec::{Vec, non_existing::*},
        //~^ ERROR: custom message
    };
}
fn main() {}
