// Resources can't be copied into other types but still need to be able
// to find their way into things.

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
    tag t {
        t0(r);
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
    // FIXME: tag constructors don't optimize their arguments into moves
    // test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
