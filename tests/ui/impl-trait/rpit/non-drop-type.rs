//@ check-pass

fn if_else(c: bool) -> impl Sized {
    if c { () } else { () }
}

fn if_no_else(c: bool) -> impl Sized {
    if c {}
}

fn matches(c: bool) -> impl Sized {
    match c {
        true => (),
        _ => (),
    }
}

fn tuple_tuple(c: bool) -> (impl Sized,) {
    if c { ((),) } else { ((),) }
}

fn main() {}
