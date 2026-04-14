//! Limine bootloader setup.

use crate::common::Result;
use xshell::{Shell, cmd};

/// Clone and build Limine bootloader if not present.
pub fn limine(sh: &Shell) -> Result<()> {
    if sh.path_exists("vendor/limine") {
        println!("Limine already present, skipping clone.");
        return Ok(());
    }

    println!("Cloning Limine v10.x...");
    sh.create_dir("vendor")?;
    cmd!(
        sh,
        "git clone https://github.com/limine-bootloader/limine.git --branch=v10.x-binary --depth=1 vendor/limine"
    )
    .run()?;
    cmd!(sh, "make -C vendor/limine").run()?;

    Ok(())
}
