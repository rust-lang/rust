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
            AllowIncoherentImpl(..) => No,
            AllowInternalUnstable(..) => Yes,
            AsPtr(..) => Yes,
            AutomaticallyDerived(..) => Yes,
            BodyStability { .. } => No,
            CoherenceIsCore => No,
            Coinductive(..) => No,
            Cold(..) => No,
            Confusables { .. } => Yes,
            ConstContinue(..) => No,
            ConstStability { .. } => Yes,
            ConstStabilityIndirect => No,
            ConstTrait(..) => No,
            Coverage(..) => No,
            DenyExplicitImpl(..) => No,
            Deprecation { .. } => Yes,
            DoNotImplementViaObject(..) => No,
            DocComment { .. } => Yes,
            Dummy => No,
            ExportName { .. } => Yes,
            ExportStable => No,
            FfiConst(..) => No,
            FfiPure(..) => No,
            Fundamental { .. } => Yes,
            Ignore { .. } => No,
            Inline(..) => No,
            LinkName { .. } => Yes,
            LinkOrdinal { .. } => No,
            LinkSection { .. } => No,
            LoopMatch(..) => No,
            MacroTransparency(..) => Yes,
            Marker(..) => No,
            MayDangle(..) => No,
            MustUse { .. } => Yes,
            Naked(..) => No,
            NoImplicitPrelude(..) => No,
            NoMangle(..) => No,
            NonExhaustive(..) => Yes,
            OmitGdbPrettyPrinterSection => No,
            Optimize(..) => No,
            ParenSugar(..) => No,
            PassByValue(..) => Yes,
            Path(..) => No,
            Pointee(..) => No,
            PubTransparent(..) => Yes,
            Repr { .. } => No,
            RustcLayoutScalarValidRangeEnd(..) => Yes,
            RustcLayoutScalarValidRangeStart(..) => Yes,
            RustcObjectLifetimeDefault => No,
            SkipDuringMethodDispatch { .. } => No,
            SpecializationTrait(..) => No,
            Stability { .. } => Yes,
            StdInternalSymbol(..) => No,
            TargetFeature(..) => No,
            TrackCaller(..) => Yes,
            TypeConst(..) => Yes,
            UnsafeSpecializationMarker(..) => No,
            UnstableFeatureBound(..) => No,
            Used { .. } => No,
            // tidy-alphabetical-end
        }
    }
}
