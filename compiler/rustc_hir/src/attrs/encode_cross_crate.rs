use crate::attrs::AttributeKind;

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
            AllowInternalUnsafe(..) => Yes,
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
            Coroutine(..) => No,
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
            LinkName { .. } => Yes, // Needed for rustdoc
            LinkOrdinal { .. } => No,
            LinkSection { .. } => Yes, // Needed for rustdoc
            LoopMatch(..) => No,
            MacroEscape(..) => No,
            MacroTransparency(..) => Yes,
            MacroUse { .. } => No,
            Marker(..) => No,
            MayDangle(..) => No,
            MustUse { .. } => Yes,
            Naked(..) => No,
            NoImplicitPrelude(..) => No,
            NoMangle(..) => Yes,      // Needed for rustdoc
            NonExhaustive(..) => Yes, // Needed for rustdoc
            Optimize(..) => No,
            ParenSugar(..) => No,
            PassByValue(..) => Yes,
            Path(..) => No,
            Pointee(..) => No,
            ProcMacro(..) => No,
            ProcMacroAttribute(..) => No,
            ProcMacroDerive { .. } => No,
            PubTransparent(..) => Yes,
            Repr { .. } => No,
            RustcBuiltinMacro { .. } => Yes,
            RustcLayoutScalarValidRangeEnd(..) => Yes,
            RustcLayoutScalarValidRangeStart(..) => Yes,
            RustcObjectLifetimeDefault => No,
            ShouldPanic { .. } => No,
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
