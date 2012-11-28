struct r {
  b: @mut int,
}

impl r : Drop {
    fn finalize(&self) {
        *(self.b) += 1;
    }
}

fn r(b: @mut int) -> r {
    r {
        b: b
    }
}

fn main() {
    let b = @mut 0;
    {
        let p = Some(r(b));
    }

    assert *b == 1;
}
