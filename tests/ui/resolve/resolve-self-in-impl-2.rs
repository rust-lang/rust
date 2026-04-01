struct S<T = u8>(T);
trait Tr<T = u8> {}

impl Self for S {} //~ ERROR expected trait, found self type `Self`
impl Self::N for S {} //~ ERROR cannot find trait `N` in `Self`

fn main() {}
