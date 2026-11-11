//@ run-pass

// A test for something that NLL enables. It sometimes happens that
// the `while let` pattern makes some borrows from a variable (in this
// case, `x`) that you need in order to compute the next value for
// `x`.  The lexical checker makes this very painful. The NLL checker
// does not.

use std::rc::Rc;

#[derive(Debug, PartialEq, Eq)]
enum Foo {
    Base(usize),
    Next(Rc<Foo>),
}

fn find_base(mut x: Rc<Foo>) -> Rc<Foo> {
    while let Foo::Next(n) = &*x {
        x = n.clone();
    }
    x
}

fn main() {
    let chain = Rc::new(Foo::Next(Rc::new(Foo::Base(44))));
    let base = find_base(chain);
    assert_eq!(&*base, &Foo::Base(44));
}
