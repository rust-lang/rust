enum foo = {mut bar: baz};

enum baz = @{mut baz: int};

trait frob {
    fn frob();
}

impl foo: frob {
    fn frob() {
        really_impure(self.bar);
    }
}

// Override default mode so that we are passing by value
fn really_impure(++bar: baz) {
    bar.baz = 3;
}

fn main() {}
