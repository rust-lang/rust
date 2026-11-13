struct Foo(isize, isize, isize, isize);

pub fn main() {
    let Self::anything_here_kills_it(a, b, ..) = Foo(5, 5, 5, 5);
    //~^ ERROR: cannot find `Self` in this scope
    match [5, 5, 5, 5] {
        [..] => {},
    }
}
