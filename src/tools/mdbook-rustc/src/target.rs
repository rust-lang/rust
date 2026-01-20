use std::borrow::Cow;

use mdbook_preprocessor::errors::Error;
use serde::Deserialize;
use serde_json::Value;

type StaticCow<T> = Cow<'static, T>;

/// This is a partial mirror of [`rustc_target::spec::Target`], containing only the relevant field(s).
/// It also has the target tuple itself added as a field for convenience.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
#[serde(rename_all = "kebab-case")]
pub struct Target {
    pub tuple: StaticCow<str>,
    pub metadata: TargetMetadata,
}

/// This is a mirror of [`rustc_target::spec::TargetMetadata`], since we can't depend on rustc_target directly.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
#[serde(deny_unknown_fields)]
pub struct TargetMetadata {
    /// A short description of the target including platform requirements,
    /// for example "64-bit Linux (kernel 3.2+, glibc 2.17+)".
    pub description: Option<StaticCow<str>>,
    /// The tier of the target. 1, 2 or 3.
    pub tier: Option<u64>,
    /// Whether the Rust project ships host tools for a target.
    pub host_tools: Option<bool>,
    /// Whether a target has the `std` library. This is usually true for targets running
    /// on an operating system.
    pub std: Option<bool>,
    /// The name of the documentation chapter for this target. Defaults to the target tuple.
    pub doc_chapter: Option<StaticCow<str>>,
}

pub struct ByTier<'a> {
    pub tier1_host: Vec<&'a Target>,
    pub tier1_nohost: Vec<&'a Target>,
    pub tier2_host: Vec<&'a Target>,
    pub tier2_nohost: Vec<&'a Target>,
    pub tier3: Vec<&'a Target>,
}

impl<'a> From<&'a [Target]> for ByTier<'a> {
    fn from(value: &'a [Target]) -> Self {
        let mut tier1_host = Vec::new();
        let mut tier1_nohost = Vec::new();
        let mut tier2_host = Vec::new();
        let mut tier2_nohost = Vec::new();
        let mut tier3 = Vec::new();

        for target in value {
            let host_tools = target.metadata.host_tools.unwrap_or(false);

            match target.metadata.tier {
                Some(1) if host_tools => tier1_host.push(target),
                Some(1) => tier1_nohost.push(target),
                Some(2) if host_tools => tier2_host.push(target),
                Some(2) => tier2_nohost.push(target),
                Some(3) => tier3.push(target),
                Some(tier) => panic!("invalid target tier for '{}': {tier:?}", target.tuple),
                None => {
                    eprintln!(
                        "WARNING: target tier not specified for '{}', defaulting to tier 3",
                        target.tuple
                    );
                    tier3.push(target);
                }
            }
        }

        Self { tier1_host, tier1_nohost, tier2_host, tier2_nohost, tier3 }
    }
}

pub fn all_from_rustc_json(json: Value) -> Result<Vec<Target>, Error> {
    let Value::Object(map) = json else {
        return Err(Error::msg("rustc didn't output a json object"));
    };
    let mut targets: Vec<Target> = Vec::with_capacity(map.len());

    for (tuple, mut target) in map {
        let target_map = target.as_object_mut().ok_or(Error::msg("target with non-object spec"))?;
        let old = target_map.insert("tuple".to_owned(), Value::String(tuple));
        assert!(old.is_none(), "rustc_target::Target does not have a 'tuple' field");
        targets.push(serde_json::from_value(target)?);
    }

    targets.sort_by(|a, b| a.tuple.cmp(&b.tuple));
    Ok(targets)
}
