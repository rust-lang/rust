//@ check-pass

fn main() {}

fn nth<I: Iterator>(iter: &mut I, step: usize) -> impl FnMut() -> Option<I::Item> + '_ {
    move || iter.nth(step)
}
