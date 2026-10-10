use std::io::Write;

pub fn hello() {
    print!("indirectly_linked.");
    std::io::stdout().flush().unwrap();
}
