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
            ExportName { .. } => Yes,
            Inline(..) => No,
            LinkSection { .. } => No,
            MacroTransparency(..) => Yes,
            Repr(..) => No,
            Stability { .. } => Yes,
            Cold(..) => No,
            ConstContinue(..) => No,
            LinkName { .. } => Yes,
            LoopMatch(..) => No,
            MayDangle(..) => No,
            MustUse { .. } => Yes,
            Naked(..) => No,
            NoMangle(..) => No,
            Optimize(..) => No,
            PubTransparent(..) => Yes,
            RustcLayoutScalarValidRangeEnd(..) => Yes,
            RustcLayoutScalarValidRangeStart(..) => Yes,
            RustcObjectLifetimeDefault => No,
            SkipDuringMethodDispatch { .. } => No,
            TrackCaller(..) => Yes,
            Used { .. } => No,
        }
    }
}
