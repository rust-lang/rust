use std::io::Write;

fn main() {
    print!("directly_linked.");
    std::io::stdout().flush().unwrap();
}
