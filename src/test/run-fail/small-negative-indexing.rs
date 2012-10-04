// error-pattern:index out of bounds: the len is 1024 but the index is -1
fn main() {
    let v = vec::from_fn(1024u, {|n| n});
    // this should trip a bounds check
    log(error, v[-1i8]);
}
