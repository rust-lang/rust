use crate::common::Result;
use clap::Args;

#[derive(Args, Clone, Debug)]
pub struct ScanArgs {
    #[arg(long)]
    pub path: Option<String>,
}

pub fn run(args: ScanArgs) -> Result<()> {
    println!("xtask: scan requested: {args:?}");
    Ok(())
}
