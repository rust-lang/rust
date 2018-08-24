// error-pattern: panic

fn main() {
    Box::new(panic!());
}
