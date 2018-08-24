enum foo { alpha, beta(isize) }

fn main() {
    match foo::alpha {
      foo::alpha | foo::beta(i) => {}
      //~^ ERROR variable `i` is not bound in all patterns
    }
}
