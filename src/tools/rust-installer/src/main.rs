use anyhow::{Context, Result};
use clap::{self, Parser};

#[derive(Parser)]
struct CommandLine {
    #[clap(subcommand)]
    command: Subcommand,
}

#[derive(clap::Subcommand)]
enum Subcommand {
    Generate(installer::Generator),
    Combine(installer::Combiner),
    Script(installer::Scripter),
    Tarball(installer::Tarballer),
}

fn main() -> Result<()> {
    let command_line = CommandLine::parse();
    match command_line.command {
        Subcommand::Combine(combiner) => combiner.run().context("failed to combine installers")?,
        Subcommand::Generate(generator) => {
            generator.run().context("failed to generate installer")?
        }
        Subcommand::Script(scripter) => {
            scripter.run().context("failed to generate installation script")?
        }
        Subcommand::Tarball(tarballer) => tarballer.run().context("failed to generate tarballs")?,
    }
    Ok(())
}
