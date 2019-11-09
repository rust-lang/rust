#[doc] //~ ERROR attribute must be of the form
//~^ WARN this was previously accepted
#[ignore()] //~ ERROR attribute must be of the form
//~^ WARN this was previously accepted
#[inline = ""] //~ ERROR attribute must be of the form
//~^ WARN this was previously accepted
#[link] //~ ERROR attribute must be of the form
//~^ WARN this was previously accepted
#[link = ""] //~ ERROR attribute must be of the form
//~^ WARN this was previously accepted

fn main() {}
