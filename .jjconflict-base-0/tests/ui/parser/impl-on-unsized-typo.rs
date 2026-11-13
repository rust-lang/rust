trait Tr {}

impl<T ?Sized> Tr for T {}
//~^ ERROR expected one of `,`, `:`, `=`, or `>`, found `?`

fn main() {}
