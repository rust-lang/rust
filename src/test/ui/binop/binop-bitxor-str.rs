// error-pattern:`^` cannot be applied to type `std::string::String`

fn main() { let x = "a".to_string() ^ "b".to_string(); }
