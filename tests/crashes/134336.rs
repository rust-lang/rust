//@ known-bug: #134336
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

trait Tr {
    fn f();
}

fn g<T: Tr>() {
    become T::f();
}
