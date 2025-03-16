#![warn(clippy::disallowed_types)]

#[allow(clippy::disallowed_types)]
mod wrapper {
    pub struct String(std::string::String);

    impl From<&str> for String {
        fn from(value: &str) -> Self {
            Self(std::string::String::from(value))
        }
    }
}

fn main() {
    let _ = String::from("x");
    //~^ disallowed_types
}
