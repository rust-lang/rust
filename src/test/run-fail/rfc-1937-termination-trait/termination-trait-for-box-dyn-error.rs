// error-pattern:returned Box<dyn Error> from main()
// failure-status: 1

use std::error::Error;
use std::io;

fn main() -> Result<(), Box<dyn Error>> {
    Err(Box::new(io::Error::new(io::ErrorKind::Other, "returned Box<dyn Error> from main()")))
}
