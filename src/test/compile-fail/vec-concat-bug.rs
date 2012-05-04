fn concat<T: copy>(v: [const [const T]]) -> [T] {
    let mut r = [];

    // Earlier versions of our type checker accepted this:
    vec::iter(v) {|&&inner: [T]|
        //!^ ERROR values differ in mutability
        r += inner;
    }

    ret r;
}

fn main() {}