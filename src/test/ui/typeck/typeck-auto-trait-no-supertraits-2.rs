#![feature(optin_builtin_traits)]

auto trait Magic : Sized where Option<Self> : Magic {} //~ ERROR E0568
impl<T:Magic> Magic for T {}

fn copy<T: Magic>(x: T) -> (T, T) { (x, x) }

#[derive(Debug)]
struct NoClone;

fn main() {
    let (a, b) = copy(NoClone);
    println!("{:?} {:?}", a, b);
}
