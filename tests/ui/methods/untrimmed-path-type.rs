// Ensures that the path of the `Error` type is not trimmed
// to make it clear which Error type is meant.

fn main() {
   meow().unknown(); //~ ERROR no method named `unknown` found
   //~^ NOTE method not found in `Result<(), std::io::Error>`
}

fn meow() -> Result<(), std::io::Error> {
    Ok(())
}
