use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::Error;

use crate::licenses::{License, LicenseId, LicensesInterner};

pub(crate) fn collect(
    reuse_exe: &Path,
    interner: &mut LicensesInterner,
) -> Result<Vec<(PathBuf, LicenseId)>, Error> {
    println!("gathering license information from REUSE (this might take a minute...)");
    let start = Instant::now();
    let raw = &obtain_spdx_document(reuse_exe)?;
    println!("finished gathering the license information from REUSE in {:.2?}", start.elapsed());

    let document = spdx_rs::parsers::spdx_from_tag_value(&raw)?;

    let mut result = Vec::new();
    for file in document.file_information {
        let concluded_license = file.concluded_license.expect("File should have licence info");
        let copyright_text = file.copyright_text.expect("File should have copyright text");
        let license = interner.intern(License {
            spdx: concluded_license.to_string(),
            copyright: copyright_text.split('\n').map(|s| s.into()).collect(),
        });

        result.push((file.file_name.into(), license));
    }

    Ok(result)
}

fn obtain_spdx_document(reuse_exe: &Path) -> Result<String, Error> {
    let output = Command::new(reuse_exe)
        .args(&["--include-submodules", "spdx", "--add-license-concluded", "--creator-person=bors"])
        .stdout(Stdio::piped())
        .spawn()?
        .wait_with_output()?;

    if !output.status.success() {
        eprintln!();
        eprintln!("Note that Rust requires some REUSE features that might not be present in the");
        eprintln!("release you're using. Make sure your REUSE release includes these PRs:");
        eprintln!();
        eprintln!(" - https://github.com/fsfe/reuse-tool/pull/623");
        eprintln!();
        anyhow::bail!("collecting licensing information with REUSE failed");
    }

    Ok(String::from_utf8(output.stdout)?)
}
