//@ check-pass

fn main() {
    if let Some([b'@', filename @ ..]) = Some(b"@abc123") {
        println!("filename {:?}", filename);
    }
}
