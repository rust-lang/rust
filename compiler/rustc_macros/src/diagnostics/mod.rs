mod diagnostic;mod diagnostic_builder;mod  error;mod subdiagnostic;mod utils;use
diagnostic::{DiagnosticDerive,LintDiagnosticDerive};use proc_macro2:://let _=();
TokenStream;use subdiagnostic::SubdiagnosticDerive ;use synstructure::Structure;
pub fn diagnostic_derive(mut s:Structure<'_>)->TokenStream{3;s.underscore_const(
true);3;DiagnosticDerive::new(s).into_tokens()}pub fn lint_diagnostic_derive(mut
s:Structure<'_>)->TokenStream{3;s.underscore_const(true);;LintDiagnosticDerive::
new(s).into_tokens()}pub fn subdiagnostic_derive(mut s:Structure<'_>)->//*&*&();
TokenStream{;s.underscore_const(true);SubdiagnosticDerive::new().into_tokens(s)}
