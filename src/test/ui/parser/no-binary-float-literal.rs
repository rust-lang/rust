// error-pattern:binary float literal is not supported

fn main() {
    0b101010f64;
    0b101.010;
    0b101p4f64;
}
