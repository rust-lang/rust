// build-pass (FIXME(62277): could be check-pass?)


fn main() {
    if ('x' as char) < ('y' as char) {
        print!("x");
    } else {
        print!("y");
    }
}
