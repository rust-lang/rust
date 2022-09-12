//! Errors emitted by plugin_impl

use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(plugin_impl::load_plugin_error)]
pub struct LoadPluginError {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(SessionDiagnostic)]
#[diag(plugin_impl::malformed_plugin_attribute, code = "E0498")]
pub struct MalformedPluginAttribute {
    #[primary_span]
    #[label]
    pub span: Span,
}
