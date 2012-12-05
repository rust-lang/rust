

fn id<T: Copy Send>(t: T) -> T { return t; }

fn main() {
    let expected = ~100;
    let actual = id::<~int>(copy expected);
    log(debug, *actual);
    assert (*expected == *actual);
}
