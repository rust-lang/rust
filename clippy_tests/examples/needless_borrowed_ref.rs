#![feature(plugin)]
#![plugin(clippy)]

#[warn(needless_borrowed_reference)]
#[allow(unused_variables)]
fn main() {
    let mut v = Vec::<String>::new();
    let _ = v.iter_mut().filter(|&ref a| a.is_empty());
    //                            ^ should be linted

    let mut var = 5;
    let thingy = Some(&mut var);
    if let Some(&mut ref v) = thingy {
        //          ^ should *not* be linted
        // here, var is borrowed as immutable.
        // can't do that:
        //*v = 10;
    }
}

#[allow(dead_code)]
enum Animal {
    Cat(u64),
    Dog(u64),
}

#[allow(unused_variables)]
#[allow(dead_code)]
fn foo(a: &Animal, b: &Animal) {
    match (a, b) {
        (&Animal::Cat(v), &ref k) | (&ref k, &Animal::Cat(v)) => (), // lifetime mismatch error if there is no '&ref'
        //                  ^    and   ^ should *not* be linted
        (&Animal::Dog(ref a), &Animal::Dog(_)) => ()
        //              ^ should *not* be linted
    }
}

