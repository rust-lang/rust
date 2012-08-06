enum foo { alpha, beta(int) }

fn main() {
    match alpha {
      alpha | beta(i) => {} //~ ERROR inconsistent number of bindings
    }
}
