
fn main() {
    let v = vec::from_fn(1024u) {|n| n};
    // this should trip a bounds check
    log(error, v[-1i8]);
}
