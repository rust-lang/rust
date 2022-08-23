// use rustc_errors::ErrorGuaranteed;
use rustc_macros::SessionDiagnostic;

#[derive(SessionDiagnostic)]
#[diag(metadata::rlib_required)]
pub struct RlibRequired {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::lib_required)]
pub struct LibRequired {
    pub crate_name: String,
    pub kind: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::crate_dep_multiple)]
#[help]
pub struct CrateDepMultiple {
    pub crate_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::two_panic_runtimes)]
pub struct TwoPanicRuntimes {
    pub prev_name: String,
    pub cur_name: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::bad_panic_strategy)]
pub struct BadPanicStrategy {
    pub runtime: String,
    pub strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::required_panic_strategy)]
pub struct RequiredPanicStrategy {
    pub crate_name: String,
    pub found_strategy: String,
    pub desired_strategy: String,
}

#[derive(SessionDiagnostic)]
#[diag(metadata::incompatible_panic_in_drop_strategy)]
pub struct IncompatiblePanicInDropStrategy {
    pub crate_name: String,
    pub found_strategy: String,
    pub desired_strategy: String,
}
