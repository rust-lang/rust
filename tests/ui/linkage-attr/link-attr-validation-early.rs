// Top-level ill-formed
#[link] //~ ERROR attribute must be of the form
        //~| WARN this was previously accepted
#[link = "foo"] //~ ERROR attribute must be of the form
                //~| WARN this was previously accepted
extern "C" {}

fn main() {}
