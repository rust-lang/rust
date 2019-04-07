// error-pattern: thread 'main' panicked at 'explicit panic'

fn main() {
    let mut vec = vec![];
    vec.push((vec.len(), panic!()));
}
