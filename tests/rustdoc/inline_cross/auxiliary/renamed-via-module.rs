#![crate_name = "foo"]

pub mod iter {
    mod range {
        pub struct StepBy;
    }
    pub use self::range::StepBy as DeprecatedStepBy;
    pub struct StepBy;
}
