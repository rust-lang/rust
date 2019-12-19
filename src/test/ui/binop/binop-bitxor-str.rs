// error-pattern:no implementation for `std::string::String ^ std::string::String`

fn main() { let x = "a".to_string() ^ "b".to_string(); }
