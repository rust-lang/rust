use super::json_parser::ArgPrep;

use crate::common::argument::MetadataDefinition;
use serde::Deserialize;
use serde_json::Value;
use std::ops::Range;

#[derive(Debug, PartialEq, Clone, Deserialize)]
pub enum Constraint {
    Equal(i64),
    Range(Range<i64>),
}

impl Constraint {
    pub fn to_range(&self) -> Range<i64> {
        match self {
            Constraint::Equal(eq) => *eq..*eq + 1,
            Constraint::Range(range) => range.clone(),
        }
    }
}

impl MetadataDefinition for Constraint {
    fn from_metadata(metadata: Option<Value>) -> Vec<Box<Self>> {
        let arg_prep: Option<ArgPrep> = metadata.and_then(|a| {
            if let Value::Object(_) = a {
                a.try_into().ok()
            } else {
                None
            }
        });
        let constraint: Option<Constraint> = arg_prep.and_then(|a| a.try_into().ok());
        vec![constraint]
            .into_iter()
            .filter_map(|a| a)
            .map(|a| Box::new(a))
            .collect()
    }
}

/// ARM-specific
impl TryFrom<ArgPrep> for Constraint {
    type Error = ();

    fn try_from(prep: ArgPrep) -> Result<Self, Self::Error> {
        let parsed_ints = match prep {
            ArgPrep::Immediate { min, max } => Ok((min, max)),
            _ => Err(()),
        };
        if let Ok((min, max)) = parsed_ints {
            if min == max {
                Ok(Constraint::Equal(min))
            } else {
                Ok(Constraint::Range(min..max + 1))
            }
        } else {
            Err(())
        }
    }
}
