use crate::attrs::AttributeKind;

#[derive(PartialEq)]
pub enum EncodeCrossCrate {
    Yes,
    No,
}

impl AttributeKind {
    /// Whether this attribute should be encoded in metadata files.
    ///
    /// If this is "Yes", then another crate can do `tcx.get_all_attrs(did)` for a did in this crate, and get the attribute.
    /// When this is No, the attribute is filtered out while encoding and other crate won't be able to observe it.
    /// This can be unexpectedly good for performance, so unless necessary for cross-crate compilation, prefer No.
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
            Coinductive(..) => No,
            Cold(..) => No,
            Confusables { .. } => Yes,
            ConstContinue(..) => No,
            ConstStability { .. } => Yes,
            ConstStabilityIndirect => No,
            ConstTrait(..) => No,
            Coroutine(..) => No,
            Coverage(..) => No,
            CrateName { .. } => No,
            CustomMir(_, _, _) => Yes,
            DebuggerVisualizer(..) => No,
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
            Link(..) => No,
            LinkName { .. } => Yes, // Needed for rustdoc
            LinkOrdinal { .. } => No,
            LinkSection { .. } => Yes, // Needed for rustdoc
            Linkage(..) => No,
            LoopMatch(..) => No,
            MacroEscape(..) => No,
            MacroExport { .. } => Yes,
            MacroTransparency(..) => Yes,
            MacroUse { .. } => No,
            Marker(..) => No,
            MayDangle(..) => No,
            MoveSizeLimit { .. } => No,
            MustUse { .. } => Yes,
            Naked(..) => No,
            NoCore(..) => No,
            NoImplicitPrelude(..) => No,
            NoMangle(..) => Yes, // Needed for rustdoc
            NoStd(..) => No,
            NonExhaustive(..) => Yes, // Needed for rustdoc
            ObjcClass { .. } => No,
            ObjcSelector { .. } => No,
            Optimize(..) => No,
            ParenSugar(..) => No,
            PassByValue(..) => Yes,
            Path(..) => No,
            PatternComplexityLimit { .. } => No,
            Pointee(..) => No,
            ProcMacro(..) => No,
            ProcMacroAttribute(..) => No,
            ProcMacroDerive { .. } => No,
            PubTransparent(..) => Yes,
            RecursionLimit { .. } => No,
            Repr { .. } => No,
            RustcBuiltinMacro { .. } => Yes,
            RustcCoherenceIsCore(..) => No,
            RustcLayoutScalarValidRangeEnd(..) => Yes,
            RustcLayoutScalarValidRangeStart(..) => Yes,
            RustcObjectLifetimeDefault => No,
            RustcSimdMonomorphizeLaneLimit(..) => Yes, // Affects layout computation, which needs to work cross-crate
            Sanitize { .. } => No,
            ShouldPanic { .. } => No,
            SkipDuringMethodDispatch { .. } => No,
            SpecializationTrait(..) => No,
            Stability { .. } => Yes,
            StdInternalSymbol(..) => No,
            TargetFeature { .. } => No,
            TrackCaller(..) => Yes,
            TypeConst(..) => Yes,
            TypeLengthLimit { .. } => No,
            UnsafeSpecializationMarker(..) => No,
            UnstableFeatureBound(..) => No,
            Used { .. } => No,
            // tidy-alphabetical-end
        }
    }
}
