macro_rules! m1 {
  ($abi: literal) => { extern $abi } //~ ERROR expected expression, found keyword `extern`
}

macro_rules! m2 {
  ($abi: expr) => { extern $abi } //~ ERROR expected expression, found keyword `extern`
}

fn main() {
    m1!(-2)
}

fn f() {
    m2!(-2)
}
