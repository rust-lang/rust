enum Wrapper {
    Wrap(i32),
}

use Wrapper::Wrap;

pub fn main() {
    let Wrap(x) = &Wrap(3);
    *x += 1; //~ ERROR cannot assign to immutable


    if let Some(x) = &Some(3) {
        *x += 1; //~ ERROR cannot assign to immutable
    } else {
        panic!();
    }

    while let Some(x) = &Some(3) {
        *x += 1; //~ ERROR cannot assign to immutable
        break;
    }
}
