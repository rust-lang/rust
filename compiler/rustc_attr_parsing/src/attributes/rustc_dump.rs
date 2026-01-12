use rustc_hir::Target;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};

use crate::attributes::prelude::Allow;
use crate::attributes::{NoArgsAttributeParser, OnDuplicate};
use crate::context::Stage;
use crate::target_checking::AllowedTargets;

pub(crate) struct RustcDumpUserArgs;

impl<S: Stage> NoArgsAttributeParser<S> for RustcDumpUserArgs {
    const PATH: &[Symbol] = &[sym::rustc_dump_user_args];
    const ON_DUPLICATE: OnDuplicate<S> = OnDuplicate::Error;
    const ALLOWED_TARGETS: AllowedTargets = AllowedTargets::AllowList(&[Allow(Target::Fn)]);
    const CREATE: fn(Span) -> AttributeKind = |_| AttributeKind::RustcDumpUserArgs;
}
