use std::fs;

use itertools::Itertools;

const CARGO_TOMLS: &[&'static str] = &[
    "Cargo.toml",
    "ci/Cargo.toml",
    "config_proc_macro/Cargo.toml",
];

pub(crate) fn runner() -> Result<(), String> {
    for (first, second) in CARGO_TOMLS.iter().tuple_windows() {
        let first_table = parse_to_toml(&first)?;
        let second_table = parse_to_toml(&second)?;
        assert_eq!(
            get_lints(first, &first_table)?,
            get_lints(second, &second_table)?,
            "Clippy configs do not match between {} and {}",
            first,
            second,
        );
    }
    Ok(())
}

fn parse_to_toml(path: &str) -> Result<toml::Table, String> {
    let toml_str = fs::read_to_string(path).map_err(|e| format!("reading {}: {}", path, e))?;
    toml_str
        .parse::<toml::Table>()
        .map_err(|e| format!("parsing {} as TOML: {}", path, e))
}

fn get_lints<'a>(path: &str, toml: &'a toml::Table) -> Result<&'a toml::Value, String> {
    toml.get("lints")
        .ok_or(format!("{} is missing key {}", path, "lints"))
        .and_then(|t| {
            t.get("clippy")
                .ok_or(format!("{} is missing key {}", path, "lints.clippy"))
        })
}
