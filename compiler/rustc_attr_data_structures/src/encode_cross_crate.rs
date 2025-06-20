use crate::AttributeKind;

#[derive(PartialEq)]
pub enum EncodeCrossCrate {
    Yes,
    No,
}

impl AttributeKind {
    pub fn encode_cross_crate(&self) -> EncodeCrossCrate {
        use AttributeKind::*;
        use EncodeCrossCrate::*;

        match self {
            Align { .. } => No,
            AllowConstFnUnstable(..) => No,
            AllowInternalUnstable(..) => Yes,
            AsPtr(..) => Yes,
            BodyStability { .. } => No,
            Confusables { .. } => Yes,
            ConstStability { .. } => Yes,
            ConstStabilityIndirect => No,
            Deprecation { .. } => Yes,
            DocComment { .. } => Yes,
            Inline(..) => No,
            MacroTransparency(..) => Yes,
            Repr(..) => No,
            Stability { .. } => Yes,
        }
    }
}
