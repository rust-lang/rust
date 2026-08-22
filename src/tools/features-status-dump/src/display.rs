use std::collections::HashMap;
use std::fmt::Display;

use tidy::features::Feature;

use crate::FeaturesStatus;

// Newtype because orphan rule

enum Source {
    Library,
    Compiler,
}

pub struct SourcedFeature {
    source: Source,
    pub feature: Feature,
}

// A new struct is defined rather than changing the old one, to keep serialization behaving the same.
pub struct NewFeaturesStatus {
    features: HashMap<String, SourcedFeature>,
    compare: for<'a, 'b> fn(
        &'a (&String, &SourcedFeature),
        &'b (&String, &SourcedFeature),
    ) -> std::cmp::Ordering,
}

impl Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let string = match self {
            Source::Library => "[Library]",
            Source::Compiler => "[Compiler]",
        };
        f.pad(string)
    }
}

impl Display for NewFeaturesStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sorted = self.features.iter().collect::<Vec<_>>();
        sorted.sort_by(self.compare);
        for (name, feature) in sorted {
            write!(f, "{:w_tag$}", feature.source, w_tag = 10)?;
            write!(f, " {:w_name$}", name, w_name = 40)?;
            write!(f, " is {:w_status$}", feature.feature.level, w_status = 10)?;
            if let Some(since) = feature.feature.since {
                write!(f, " since {:w_since$}", since, w_since = 7)?;
            }
            if let Some(issue) = &feature.feature.tracking_issue {
                let link = format!("<https://github.com/rust-lang/rust/issues/{}>", issue);
                write!(f, " {:<w_issue$}", link, w_issue = 49)?;
            }
            if let Some(description) = &feature.feature.description {
                write!(f, ": {}", description)?;
            }
            writeln!(f)?;
        }
        std::fmt::Result::Ok(())
    }
}

impl NewFeaturesStatus {
    pub fn new(
        value: FeaturesStatus,
        compare: for<'a, 'b> fn(
            &'a (&String, &SourcedFeature),
            &'b (&String, &SourcedFeature),
        ) -> std::cmp::Ordering,
    ) -> Self {
        let compiler_features = value
            .lang_features_status
            .into_iter()
            .map(|(name, feature)| (name, SourcedFeature { feature, source: Source::Compiler }));
        let library_features = value
            .lib_features_status
            .into_iter()
            .map(|(name, feature)| (name, SourcedFeature { feature, source: Source::Library }));
        let features = compiler_features.chain(library_features).collect();

        Self { features, compare }
    }
}
