// Ensure that both `Box<dyn Error + Send + Sync>` and `Box<dyn Error>` can be
// obtained from `String`.

use std::error::Error;

fn main() {
    let _err1: Box<dyn Error + Send + Sync> = From::from("test".to_string());
    let _err2: Box<dyn Error> = From::from("test".to_string());
    let _err3: Box<dyn Error + Send + Sync + 'static> = From::from("test");
    let _err4: Box<dyn Error> = From::from("test");
}
