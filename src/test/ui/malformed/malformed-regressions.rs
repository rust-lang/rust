// compile-pass

#[doc] //~ WARN attribute must be of the form
#[ignore()] //~ WARN attribute must be of the form
#[inline = ""] //~ WARN attribute must be of the form
#[link] //~ WARN attribute must be of the form
#[link = ""] //~ WARN attribute must be of the form
fn main() {}
