// Resources can't be copied, but storing into data structures counts
// as a move unless the stored thing is used afterwards.

resource r(i: @mutable int) {
    *i = *i + 1;
}

fn test_box() {
    let i = @mutable 0;
    {
        let a <- @r(i);
    }
    assert *i == 1;
}

fn test_rec() {
    let i = @mutable 0;
    {
        let a <- {x: r(i)};
    }
    assert *i == 1;
}

fn test_tag() {
    enum t {
        t0(r),
    }

    let i = @mutable 0;
    {
        let a <- t0(r(i));
    }
    assert *i == 1;
}

fn test_tup() {
    let i = @mutable 0;
    {
        let a <- (r(i), 0);
    }
    assert *i == 1;
}

fn test_unique() {
    let i = @mutable 0;
    {
        let a <- ~r(i);
    }
    assert *i == 1;
}

fn test_box_rec() {
    let i = @mutable 0;
    {
        let a <- @{
            x: r(i)
        };
    }
    assert *i == 1;
}

fn main() {
    test_box();
    test_rec();
    // FIXME: enum constructors don't optimize their arguments into moves
    // test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
