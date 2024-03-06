//@ run-pass
enum Wrapper {
    Wrap(i32),
}

use Wrapper::Wrap;

pub fn main() {
    let Wrap(x) = &Wrap(3);
    println!("{}", *x);

    let Wrap(x) = &mut Wrap(3);
    println!("{}", *x);

    if let Some(x) = &Some(3) {
        println!("{}", *x);
    } else {
        panic!();
    }

    if let Some(x) = &mut Some(3) {
        println!("{}", *x);
    } else {
        panic!();
    }

    if let Some(x) = &mut Some(3) {
        *x += 1;
    } else {
        panic!();
    }

    while let Some(x) = &Some(3) {
        println!("{}", *x);
        break;
    }
    while let Some(x) = &mut Some(3) {
        println!("{}", *x);
        break;
    }
    while let Some(x) = &mut Some(3) {
        *x += 1;
        break;
    }
}
