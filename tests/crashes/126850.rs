//@ known-bug: rust-lang/rust#126850
fn bug<T>() -> impl Iterator<
    Item = [(); {
               |found: &String| Some(false);
               4
           }],
> {
    std::iter::empty()
}

fn main() {}
