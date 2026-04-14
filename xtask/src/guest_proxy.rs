use crate::common::Result;

pub fn run(port: u16) -> Result<()> {
    println!("xtask: guest proxy requested on port {port}");
    Ok(())
}
