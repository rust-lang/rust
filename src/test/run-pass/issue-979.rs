struct r {
  let b: @mut int;
  new(b: @mut int) {
    self.b = b;
  }
  drop { *(self.b) += 1; }
}

fn main() {
    let b = @mut 0;
    {
        let p = Some(r(b));
    }

    assert *b == 1;
}