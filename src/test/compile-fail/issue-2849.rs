enum foo { alpha, beta(int) }

fn main() {
    match alpha {
      alpha | beta(i) => {} //~ ERROR variable `i` from pattern #2 is not bound in pattern #1
    }
}
