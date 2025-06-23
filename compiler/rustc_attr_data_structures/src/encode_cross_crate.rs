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
            // tidy-alphabetical-start
            Align { .. } => No,
            AllowConstFnUnstable(..) => No,
            AllowInternalUnstable(..) => Yes,
            AsPtr(..) => Yes,
            BodyStability { .. } => No,
            Cold(..) => No,
            Confusables { .. } => Yes,
            ConstContinue(..) => No,
            ConstStability { .. } => Yes,
            ConstStabilityIndirect => No,
            Deprecation { .. } => Yes,
            DocComment { .. } => Yes,
            ExportName { .. } => Yes,
            ExportStable => No,
            FfiConst(..) => No,
            Ignore { .. } => No,
            Inline(..) => No,
            LinkName { .. } => Yes,
            LinkSection { .. } => No,
            LoopMatch(..) => No,
            MacroTransparency(..) => Yes,
            MayDangle(..) => No,
            MustUse { .. } => Yes,
            Naked(..) => No,
            NoImplicitPrelude(..) => No,
            NoMangle(..) => No,
            NonExhaustive(..) => Yes,
            Optimize(..) => No,
            PassByValue(..) => Yes,
            Path(..) => No,
            PubTransparent(..) => Yes,
            Repr { .. } => No,
            RustcLayoutScalarValidRangeEnd(..) => Yes,
            RustcLayoutScalarValidRangeStart(..) => Yes,
            RustcObjectLifetimeDefault => No,
            SkipDuringMethodDispatch { .. } => No,
            Stability { .. } => Yes,
            TargetFeature(..) => No,
            TrackCaller(..) => Yes,
            Used { .. } => No,
            // tidy-alphabetical-end
        }
    }
}
