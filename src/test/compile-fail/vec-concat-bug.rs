fn concat<T: copy>(v: [const [const T]]) -> [T] {
    let mut r = [];

    // Earlier versions of our type checker accepted this:
    for inner: [T] in v {
        //!^ ERROR found `[const 'a]` (values differ in mutability)
        r += inner;
    }

    ret r;
}

fn main() {}