struct r {
  let i: @mut int;
  drop { *(self.i) = *(self.i) + 1; }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

fn main() {
    let i = @mut 0;
    {
        let j = ~r(i);
    }
    assert *i == 1;
}