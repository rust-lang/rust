//@ error-pattern:cannot apply unary operator `-` to type `String`

fn main() { -"foo".to_string(); }
