//@ check-pass

fn main() {}

#[cfg(FALSE)]
impl X {
    type Y;
    type Z: Ord;
    type W: Ord where Self: Eq;
    type W where Self: Eq;
}
