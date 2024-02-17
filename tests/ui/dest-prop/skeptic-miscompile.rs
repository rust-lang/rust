//@ run-pass

//@ compile-flags: -Zmir-opt-level=3

trait IterExt: Iterator {
    fn fold_ex<B, F>(mut self, init: B, mut f: F) -> B
    where
        Self: Sized,
        F: FnMut(B, Self::Item) -> B,
    {
        let mut accum = init;
        while let Some(x) = self.next() {
            accum = f(accum, x);
        }
        accum
    }
}

impl<T: Iterator> IterExt for T {}

fn main() {
    let test = &["\n"];
    test.iter().fold_ex(String::new(), |_, b| b.to_string());
}
