use crate::common::Result;
use xshell::Shell;

pub fn bdd(
    _sh: &Shell,
    feature: Option<String>,
    tags: Option<String>,
    arch: Vec<String>,
) -> Result<()> {
    println!("xtask: bdd feature={feature:?} tags={tags:?} arch={arch:?}");
    Ok(())
}
