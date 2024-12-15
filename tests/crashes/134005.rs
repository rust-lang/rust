//@ known-bug: #134005

fn main() {
    let _ = [std::ops::Add::add, std::ops::Mul::mul, main as fn(_, &_)];
}
