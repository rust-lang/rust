// run-pass
#![allow(non_upper_case_globals)]


pub fn main() {
    let _foo = 100;
    const quux: isize = 5;

    enum Stuff {
        Bar = quux
    }

    assert_eq!(Stuff::Bar as isize, quux);
}
