use std::io::Write;

fn main() {
    let mut w = Vec::new();
    write!(&mut w, "test")?;
    Ok(())
}
