use std::cmp::Ordering;
// For behaviour, see parse.rs
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter};

use anyhow::{Context, Result};
use tidy::diagnostics::RunningCheck;
use tidy::features::{Feature, Status, collect_lang_features, collect_lib_features};

use crate::display::{NewFeaturesStatus, SourcedFeature};
use crate::parse::{Args, Tristate};

mod display;
mod err;
mod parse;

// Placing this into a structure makes it easier to serialize
#[derive(Debug, serde::Serialize)]
struct FeaturesStatus {
    pub(crate) lang_features_status: HashMap<String, Feature>,
    pub(crate) lib_features_status: HashMap<String, Feature>,
}

fn main() -> Result<()> {
    let args = Args::parse()?;

    let lang_features_status: HashMap<_, _> = args
        .compiler_path
        .iter()
        .flat_map(|compiler_path| {
            collect_lang_features(&compiler_path, &mut RunningCheck::new_noop())
        })
        .filter(|(_, feature)| include(feature, &args))
        .collect();

    let lib_features_status = args
        .library_path
        .iter()
        .flat_map(|library_path| collect_lib_features(&library_path).into_iter())
        // The library contains less info on their features. Prefer the ones found in the compiler.
        .filter(|&(ref name, _)| !lang_features_status.contains_key(name))
        .filter(|(_, feature)| include(feature, &args))
        .collect();

    let features_status = FeaturesStatus { lang_features_status, lib_features_status };

    match &args.output_path {
        Some(output_path) => {
            let output_dir = output_path.parent().with_context(|| {
                format!("failed to get parent dir of output path `{}`", output_path.display())
            })?;
            std::fs::create_dir_all(output_dir).with_context(|| {
                format!("failed to create output directory at `{}`", output_dir.display())
            })?;

            let output_file = File::create(&output_path).with_context(|| {
                format!("failed to create file at given output path `{}`", output_path.display())
            })?;
            let writer = BufWriter::new(output_file);
            write_output(writer, features_status, &args)?;
        }
        None => {
            let writer = BufWriter::new(std::io::stdout());
            write_output(writer, features_status, &args)?;
        }
    };
    Ok(())
}

fn write_output<W>(mut writer: W, features_status: FeaturesStatus, args: &Args) -> Result<()>
where
    W: io::Write,
{
    match args.format {
        parse::Format::JSON => serde_json::to_writer_pretty(writer, &features_status)
            .context("failed to write json output"),
        parse::Format::Text => {
            let compare: for<'a, 'b> fn(
                &'a (&String, &SourcedFeature),
                &'b (&String, &SourcedFeature),
            ) -> Ordering = match args.sort_by {
                parse::SortBy::Newest => |a, b| compare_ascending(a, b).reverse(),
                parse::SortBy::Oldest => compare_ascending,
            };
            let new_features = NewFeaturesStatus::new(features_status, compare);

            write!(writer, "{}", new_features).map_err(Into::into)
        }
    }
}

fn compare_ascending<'a, 'b>(
    a: &'a (&String, &SourcedFeature),
    b: &'b (&String, &SourcedFeature),
) -> Ordering {
    match a.1.feature.since.cmp(&b.1.feature.since) {
        Ordering::Equal => (),
        other => return other,
    };
    a.1.feature.tracking_issue.cmp(&b.1.feature.tracking_issue)
}

fn include(feature: &Feature, args: &Args) -> bool {
    let accept = match args.accepted {
        Tristate::Require => feature.level == Status::Accepted,
        Tristate::Allow => true,
        Tristate::Deny => feature.level != Status::Accepted,
    };
    let remove = match args.removed {
        Tristate::Require => feature.level == Status::Removed,
        Tristate::Allow => true,
        Tristate::Deny => feature.level != Status::Removed,
    };
    let unstable = match args.unstable {
        Tristate::Require => feature.level == Status::Unstable,
        Tristate::Allow => true,
        Tristate::Deny => feature.level != Status::Unstable,
    };
    let tracking = match args.tracking_issue {
        Tristate::Require => feature.tracking_issue.is_some(),
        Tristate::Allow => true,
        Tristate::Deny => feature.tracking_issue.is_none(),
    };
    let before = match args.before {
        Some(before) => match feature.since {
            Some(version) => version < before,
            None => false, // reject features without version
        },
        None => true,
    };
    let since = match args.since {
        Some(since) => match feature.since {
            Some(version) => version >= since,
            None => false, // reject features without version
        },
        None => true,
    };
    accept && remove && unstable && tracking && before && since
}
