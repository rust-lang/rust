struct AStruct {
    A: u32,
    B: u32,
    C: u32,
}

impl Something for AStruct {
    fn a_func() {
        match a_val {
            ContextualParseError::InvalidMediaRule(ref err) => {
                let err: &CStr = match err.kind {
                    ParseErrorKind::Custom(StyleParseErrorKind::MediaQueryExpectedFeatureName(
                        ..,
                    )) => {
                        cstr!("PEMQExpectedFeatureName")
                    }
                };
            }
        };
    }
}
