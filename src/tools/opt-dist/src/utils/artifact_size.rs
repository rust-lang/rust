use std::io::Write;

use tabled::builder::Builder;
use tabled::settings::object::Columns;
use tabled::settings::style::{LineChar, Offset};
use tabled::settings::{Modify, Style};

use crate::environment::Environment;
use crate::utils::io::get_files_from_dir;

pub fn print_binary_sizes(env: &Environment) -> anyhow::Result<()> {
    use std::fmt::Write;

    use humansize::{BINARY, format_size};

    let root = env.build_artifacts().join("stage2");

    let mut files = get_files_from_dir(&root.join("bin"), None)?;
    files.extend(get_files_from_dir(&root.join("lib"), None)?);
    files.sort_unstable();

    let items: Vec<_> = files
        .into_iter()
        .map(|file| {
            let size = std::fs::metadata(file.as_std_path()).map(|m| m.len()).unwrap_or(0);
            let size_formatted = format_size(size, BINARY);
            let name = file.file_name().unwrap().to_string();
            (name, size_formatted)
        })
        .collect();

    // Write to log
    let mut output = String::new();
    for (name, size_formatted) in items.iter() {
        let name = format!("{}:", name);
        writeln!(output, "{name:<50}{size_formatted:>10}")?;
    }
    log::info!("Rustc artifact size\n{output}");

    // Write to GitHub summary
    if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        let mut builder = Builder::default();
        builder.push_record(vec!["Artifact", "Size"]);
        for (name, size_formatted) in items {
            builder.push_record(vec![name, size_formatted]);
        }

        let mut table = builder.build();

        let mut file = std::fs::File::options().append(true).create(true).open(summary_path)?;
        writeln!(
            file,
            "# Artifact size\n{}\n",
            table.with(Style::markdown()).with(
                Modify::new(Columns::single(1)).with(LineChar::horizontal(':', Offset::End(0))),
            )
        )?;
    }

    Ok(())
}
