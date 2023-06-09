// check-pass

struct Foo { a: isize, b: isize }

fn main() {
    let mut x: Box<_> = Box::new(Foo { a: 1, b: 2 });
    let (a, b) = (&mut x.a, &mut x.b);

    let mut foo: Box<_> = Box::new(Foo { a: 1, b: 2 });
    let (c, d) = (&mut foo.a, &foo.b);

    // We explicitly use the references created above to illustrate that the
    // borrow checker is accepting this code *not* because of artificially
    // short lifetimes, but rather because it understands that all the
    // references are of disjoint parts of memory.
    use_imm(d);
    use_mut(c);
    use_mut(b);
    use_mut(a);
}

fn use_mut<T>(_: &mut T) { }
fn use_imm<T>(_: &T) { }
