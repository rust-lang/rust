// compile-pass

use std::iter::Iterator;

type Unit = ();

fn test() ->  Box<Iterator<Item = (), Item = Unit>> {
    Box::new(None.into_iter())
}

fn main() {
    test();
}
