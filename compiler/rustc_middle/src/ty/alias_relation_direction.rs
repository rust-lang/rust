use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, Debug)]
pub enum AliasRelationDirection {
    Equate,
    Subtype,
}

impl fmt::Display for AliasRelationDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AliasRelationDirection::Equate => write!(f, "=="),
            AliasRelationDirection::Subtype => write!(f, "<:"),
        }
    }
}
