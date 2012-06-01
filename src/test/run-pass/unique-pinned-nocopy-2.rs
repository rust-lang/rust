class r {
  let i: @mut int;
  new(i: @mut int) { self.i = i; }
  drop { *(self.i) = *(self.i) + 1; }
}

fn main() {
    let i = @mut 0;
    {
        let j = ~r(i);
    }
    assert *i == 1;
}