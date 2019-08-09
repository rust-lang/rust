enum Wrapper {
    Wrap(i32),
}

use Wrapper::Wrap;

pub fn main() {
    let Wrap(x) = &Wrap(3);
    *x += 1; //~ ERROR cannot assign to `*x` which is behind a `&` reference


    if let Some(x) = &Some(3) {
        *x += 1; //~ ERROR cannot assign to `*x` which is behind a `&` reference
    } else {
        panic!();
    }

    while let Some(x) = &Some(3) {
        *x += 1; //~ ERROR cannot assign to `*x` which is behind a `&` reference
        break;
    }
}
