use rustc_errors::{struct_span_err, Applicability, DiagnosticBuilder};
use rustc_hir::def::DefKind;
use rustc_middle::bug;
use rustc_span::symbol::{Ident, Symbol};
use rustc_span::{BytePos, MultiSpan, Span};

use crate::{BindingError, HasGenericParams};
use crate::{ResolutionError, Resolver};

use crate::diagnostics::{reduce_impl_span_to_impl_keyword, LabelSuggestion, Res, Suggestion};
