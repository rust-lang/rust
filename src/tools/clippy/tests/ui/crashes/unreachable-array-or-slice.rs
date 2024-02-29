struct Foo(isize, isize, isize, isize);

pub fn main() {
    let Self::anything_here_kills_it(a, b, ..) = Foo(5, 5, 5, 5);
    match [5, 5, 5, 5] {
        [..] => { }
    }
}
