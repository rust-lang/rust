// Resources can't be copied, but storing into data structures counts
// as a move unless the stored thing is used afterwards.

class r {
  let i: @mut int;
  new(i: @mut int) {
    self.i = i;
  }
  drop { *(self.i) = *(self.i) + 1; }
}

fn test_box() {
    let i = @mut 0;
    {
        let a <- @r(i);
    }
    assert *i == 1;
}

fn test_rec() {
    let i = @mut 0;
    {
        let a <- {x: r(i)};
    }
    assert *i == 1;
}

fn test_tag() {
    enum t {
        t0(r),
    }

    let i = @mut 0;
    {
        let a <- t0(r(i));
    }
    assert *i == 1;
}

fn test_tup() {
    let i = @mut 0;
    {
        let a <- (r(i), 0);
    }
    assert *i == 1;
}

fn test_unique() {
    let i = @mut 0;
    {
        let a <- ~r(i);
    }
    assert *i == 1;
}

fn test_box_rec() {
    let i = @mut 0;
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
    test_tag();
    test_tup();
    test_unique();
    test_box_rec();
}
