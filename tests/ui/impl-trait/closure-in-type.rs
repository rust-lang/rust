//@ check-pass

fn bug<T>() -> impl Iterator<
    Item = [(); {
               |found: &String| Some(false);
               4
           }],
> {
    std::iter::empty()
}

fn main() {}
