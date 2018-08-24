// error-pattern:panicked at 'Box<Any>'

fn main() {
    panic!(Box::new(612_i64));
}
