#[doc] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[ignore()] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[inline = ""] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[link] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted
#[link = ""] //~ ERROR valid forms for the attribute are
//~^ WARN this was previously accepted

fn main() {}
