// Ensure that both `Box<dyn Error + Send + Sync>` and `Box<dyn Error>` can be
// obtained from `String`.

use std::error::Error;

fn main() {
    let _err1: Box<Error + Send + Sync> = From::from("test".to_string());
    let _err2: Box<Error> = From::from("test".to_string());
    let _err3: Box<Error + Send + Sync + 'static> = From::from("test");
    let _err4: Box<Error> = From::from("test");
}
