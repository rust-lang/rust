// run-rustfix

#![warn(clippy::needless_borrowed_reference)]
#![allow(unused, clippy::needless_borrow)]

fn main() {}

fn should_lint(array: [u8; 4], slice: &[u8], slice_of_refs: &[&u8], vec: Vec<u8>) {
    let mut v = Vec::<String>::new();
    let _ = v.iter_mut().filter(|&ref a| a.is_empty());

    let var = 3;
    let thingy = Some(&var);
    if let Some(&ref v) = thingy {}

    if let &[&ref a, ref b] = slice_of_refs {}

    let &[ref a, ..] = &array;
    let &[ref a, ref b, ..] = &array;

    if let &[ref a, ref b] = slice {}
    if let &[ref a, ref b] = &vec[..] {}

    if let &[ref a, ref b, ..] = slice {}
    if let &[ref a, .., ref b] = slice {}
    if let &[.., ref a, ref b] = slice {}
}

fn should_not_lint(array: [u8; 4], slice: &[u8], slice_of_refs: &[&u8], vec: Vec<u8>) {
    if let [ref a] = slice {}
    if let &[ref a, b] = slice {}
    if let &[ref a, .., b] = slice {}

    // must not be removed as variables must be bound consistently across | patterns
    if let (&[ref a], _) | ([], ref a) = (slice_of_refs, &1u8) {}

    let mut var2 = 5;
    let thingy2 = Some(&mut var2);
    if let Some(&mut ref mut v) = thingy2 {
        //          ^ should **not** be linted
        // v is borrowed as mutable.
        *v = 10;
    }
    if let Some(&mut ref v) = thingy2 {
        //          ^ should **not** be linted
        // here, v is borrowed as immutable.
        // can't do that:
        //*v = 15;
    }
}

enum Animal {
    Cat(u64),
    Dog(u64),
}

fn foo(a: &Animal, b: &Animal) {
    match (a, b) {
        // lifetime mismatch error if there is no '&ref' before `feature(nll)` stabilization in 1.63
        (&Animal::Cat(v), &ref k) | (&ref k, &Animal::Cat(v)) => (),
        //                  ^    and   ^ should **not** be linted
        (&Animal::Dog(ref a), &Animal::Dog(_)) => (), //              ^ should **not** be linted
    }
}
