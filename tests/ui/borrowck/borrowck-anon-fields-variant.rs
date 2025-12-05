enum Foo {
    X, Y(usize, usize)
}

fn distinct_variant() {
    let mut y = Foo::Y(1, 2);

    let a = match y {
      Foo::Y(ref mut a, _) => a,
      Foo::X => panic!()
    };

    // While `a` and `b` are disjoint, borrowck doesn't know that `a` is not
    // also used for the discriminant of `Foo`, which it would be if `a` was a
    // reference.
    let b = match y {
      //~^ ERROR cannot use `y`
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
      //~^ ERROR cannot use `y`
      Foo::Y(ref mut b, _) => b,
      //~^ ERROR cannot borrow `y.0` as mutable
      Foo::X => panic!()
    };

    *a += 1;
    *b += 1;
}

fn main() {
}
