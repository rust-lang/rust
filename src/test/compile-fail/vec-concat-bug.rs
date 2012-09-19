fn concat<T: Copy>(v: ~[const ~[const T]]) -> ~[T] {
    let mut r = ~[];

    // Earlier versions of our type checker accepted this:
    vec::each(v, |inner: &~[T]| {
        //~^ ERROR values differ in mutability
        r += *inner; true
    });

    return r;
}

fn main() {}