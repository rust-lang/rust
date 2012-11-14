struct r {
  i: @mut int,
}

impl r : Drop {
    fn finalize() {
        *(self.i) += 1;
    }
}

fn r(i: @mut int) -> r {
    r {
        i: i
    }
}

fn main() {
    let i = @mut 0;
    // Even though these look like copies, they are guaranteed not to be
    {
        let a = r(i);
        let b = (move a, 10);
        let (c, _d) = move b;
        log(debug, c);
    }
    assert *i == 1;
}
