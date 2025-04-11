macro_rules! m {
  ($abi : expr) => { extern $abi } //~ ERROR expected expression, found keyword `extern`
}

fn main() {
    m!(-2)
}
