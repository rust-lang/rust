// OIBIT-based version of #29859, supertrait version. Test that using
// a simple OIBIT `..` impl alone still doesn't allow arbitrary bounds
// to be synthesized.

#![feature(optin_builtin_traits)]

auto trait Magic: Copy {} //~ ERROR E0568

fn copy<T: Magic>(x: T) -> (T, T) { (x, x) }

#[derive(Debug)]
struct NoClone;

fn main() {
    let (a, b) = copy(NoClone); //~ ERROR
    println!("{:?} {:?}", a, b);
}
