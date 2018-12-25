// error-pattern:cannot apply unary operator `-` to type `std::string::String`

fn main() { -"foo".to_string(); }
