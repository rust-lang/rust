enum foo { alpha, beta(int) }

fn main() {
    alt alpha {
      alpha | beta(i) => {} //~ ERROR inconsistent number of bindings
    }
}
