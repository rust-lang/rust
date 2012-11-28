// Resources can't be copied, but storing into data structures counts
// as a move unless the stored thing is used afterwards.

struct r {
  i: @mut int,
}

impl r : Drop {
    fn finalize(&self) {
        *(self.i) = *(self.i) + 1;
    }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

fn test_box() {
    let i = @mut 0;
    {
        let a = move @r(i);
    }
    assert *i == 1;
}

fn test_rec() {
    let i = @mut 0;
    {
        let a = move {x: r(i)};
    }
    assert *i == 1;
}

fn test_tag() {
    enum t {
        t0(r),
    }

    let i = @mut 0;
    {
        let a = move t0(r(i));
    }
    assert *i == 1;
}

fn test_tup() {
    let i = @mut 0;
    {
        let a = move (r(i), 0);
    }
    assert *i == 1;
}

fn test_unique() {
    let i = @mut 0;
    {
        let a = move ~r(i);
    }
    assert *i == 1;
}

fn test_box_rec() {
    let i = @mut 0;
    {
        let a = move @{
            x: r(i)
        };
    }
    assert *i == 1;
}

fn main() {
    test_box();
    test_rec();
    test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
