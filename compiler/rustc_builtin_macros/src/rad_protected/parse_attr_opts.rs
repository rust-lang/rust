use rustc_ast as ast;
use rustc_ast::MetaItemInner;
use rustc_expand::base::ExtCtxt;
use thin_vec::{thin_vec, ThinVec};
use rustc_span::sym;

pub(super) struct AttrOpts {
    triplicate_unsafe: bool,
    shadow_only: bool,
}

impl AttrOpts {
    pub(super) fn triplicate_unsafe(&self) -> bool {
        self.triplicate_unsafe
    }

    /// `#[rad_protected(shadow_only)]`: run only the speculative-shadow MIR pass, without
    /// triplicating the process or injecting checkpoints. Intended for testing that pass alone.
    pub(super) fn shadow_only(&self) -> bool {
        self.shadow_only
    }
}

pub(super) fn parse_attr_opts(cx: &ExtCtxt<'_>, meta_item: &ast::MetaItem) -> Option<AttrOpts> {

    let attr_opts: ThinVec<MetaItemInner> = match meta_item.kind {
        ast::MetaItemKind::List(ref vec) => vec.clone(),
        ast::MetaItemKind::Word => thin_vec![],
        _ => {
            cx.dcx().span_err(meta_item.span, "unsupported options kind in `#[rad_protected]`");
            return None;
        }
    };

    let mut triplicate_unsafe = false;
    let mut shadow_only = false;

    for opt in attr_opts {
        match opt {
            MetaItemInner::MetaItem(opt) if opt.has_name(sym::triplicate_unsafe) => {
                triplicate_unsafe = true;
            }
            MetaItemInner::MetaItem(opt) if opt.has_name(sym::shadow_only) => {
                shadow_only = true;
            }
            _ => {
                cx.dcx().span_err(opt.span(), "unsupported option in `#[rad_protected]`");
               return None;
            }
        }
    }

    Some(AttrOpts { triplicate_unsafe, shadow_only })
}
