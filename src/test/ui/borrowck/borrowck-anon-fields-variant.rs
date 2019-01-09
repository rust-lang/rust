// Tests that we are able to distinguish when loans borrow different
// anonymous fields of an enum variant vs the same anonymous field.

enum Foo {
    X, Y(usize, usize)
}

fn distinct_variant() {
    let mut y = Foo::Y(1, 2);

    let a = match y {
      Foo::Y(ref mut a, _) => a,
      Foo::X => panic!()
    };

    let b = match y {
      Foo::Y(_, ref mut b) => b,
      Foo::X => panic!()
    };

    *a += 1;
    *b += 1;
}

fn same_variant() {
    let mut y = Foo::Y(1, 2);

    let a = match y {
      Foo::Y(ref mut a, _) => a,
      Foo::X => panic!()
    };

    let b = match y {
      Foo::Y(ref mut b, _) => b, //~ ERROR cannot borrow
      Foo::X => panic!()
    };

    *a += 1;
    *b += 1;
}

fn main() {
}
