#[derive(Debug)]
enum MyError {
    NotFound { key: Vec<u8> },
    Err41,
}

impl std::error::Error for MyError {}

impl std::fmt::Display for MyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            MyError::NotFound { key } => write!(
                f,
                "unknown error with code {}.",
                String::from_utf8(*key).unwrap()
                //~^ ERROR cannot move out of `*key` which is behind a shared reference
            ),
            MyError::Err41 => write!(f, "Sit by a lake"),
        }
    }
}
fn main() {
    println!("Hello, world!");
}
