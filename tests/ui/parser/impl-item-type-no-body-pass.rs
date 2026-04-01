//@ check-pass

fn main() {}

#[cfg(false)]
impl X {
    type Y;
    type Z: Ord;
    type W: Ord where Self: Eq;
    type W where Self: Eq;
}
