//
// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    let mut i = [1, 2, 3];
    i[i[0]] = 0;
    i[i[0] - 1] = 0;
}
