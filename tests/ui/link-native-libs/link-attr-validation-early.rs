// Top-level ill-formed
#[link] //~ ERROR valid forms for the attribute are
        //~| WARN this was previously accepted
#[link = "foo"] //~ ERROR valid forms for the attribute are
                //~| WARN this was previously accepted
extern "C" {}

fn main() {}
