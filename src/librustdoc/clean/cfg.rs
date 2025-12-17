//! The representation of a `#[doc(cfg(...))]` attribute.

// FIXME: Once the portability lint RFC is implemented (see tracking issue #41619),
// switch to use those structures instead.

use std::sync::Arc;
use std::{fmt, mem, ops};

use itertools::Either;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::thin_vec::{ThinVec, thin_vec};
use rustc_hir as hir;
use rustc_hir::Attribute;
use rustc_hir::attrs::{self, AttributeKind, CfgEntry, CfgHideShow, HideOrShow};
use rustc_middle::ty::TyCtxt;
use rustc_span::symbol::{Symbol, sym};
use rustc_span::{DUMMY_SP, Span};

use crate::display::{Joined as _, MaybeDisplay, Wrapped};
use crate::html::escape::Escape;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Hash)]
// Because `CfgEntry` includes `Span`, we must NEVER use `==`/`!=` operators on `Cfg` and instead
// use `is_equivalent_to`.
#[cfg_attr(test, derive(PartialEq))]
pub(crate) struct Cfg(CfgEntry);

/// Whether the configuration consists of just `Cfg` or `Not`.
fn is_simple_cfg(cfg: &CfgEntry) -> bool {
    match cfg {
        CfgEntry::Bool(..)
        | CfgEntry::NameValue { .. }
        | CfgEntry::Not(..)
        | CfgEntry::Version(..) => true,
        CfgEntry::All(..) | CfgEntry::Any(..) => false,
    }
}

/// Returns `false` if is `Any`, otherwise returns `true`.
fn is_all_cfg(cfg: &CfgEntry) -> bool {
    match cfg {
        CfgEntry::Bool(..)
        | CfgEntry::NameValue { .. }
        | CfgEntry::Not(..)
        | CfgEntry::Version(..)
        | CfgEntry::All(..) => true,
        CfgEntry::Any(..) => false,
    }
}

fn strip_hidden(cfg: &CfgEntry, hidden: &FxHashSet<NameValueCfg>) -> Option<CfgEntry> {
    match cfg {
        CfgEntry::Bool(..) => Some(cfg.clone()),
        CfgEntry::NameValue { .. } => {
            if !hidden.contains(&NameValueCfg::from(cfg)) {
                Some(cfg.clone())
            } else {
                None
            }
        }
        CfgEntry::Not(cfg, _) => {
            if let Some(cfg) = strip_hidden(cfg, hidden) {
                Some(CfgEntry::Not(Box::new(cfg), DUMMY_SP))
            } else {
                None
            }
        }
        CfgEntry::Any(cfgs, _) => {
            let cfgs =
                cfgs.iter().filter_map(|cfg| strip_hidden(cfg, hidden)).collect::<ThinVec<_>>();
            if cfgs.is_empty() { None } else { Some(CfgEntry::Any(cfgs, DUMMY_SP)) }
        }
        CfgEntry::All(cfgs, _) => {
            let cfgs =
                cfgs.iter().filter_map(|cfg| strip_hidden(cfg, hidden)).collect::<ThinVec<_>>();
            if cfgs.is_empty() { None } else { Some(CfgEntry::All(cfgs, DUMMY_SP)) }
        }
        CfgEntry::Version(..) => {
            // FIXME: Should be handled.
            Some(cfg.clone())
        }
    }
}

fn should_capitalize_first_letter(cfg: &CfgEntry) -> bool {
    match cfg {
        CfgEntry::Bool(..) | CfgEntry::Not(..) | CfgEntry::Version(..) => true,
        CfgEntry::Any(sub_cfgs, _) | CfgEntry::All(sub_cfgs, _) => {
            sub_cfgs.first().map(should_capitalize_first_letter).unwrap_or(false)
        }
        CfgEntry::NameValue { name, .. } => {
            *name == sym::debug_assertions || *name == sym::target_endian
        }
    }
}

impl Cfg {
    /// Renders the configuration for human display, as a short HTML description.
    pub(crate) fn render_short_html(&self) -> String {
        let mut msg = Display(&self.0, Format::ShortHtml).to_string();
        if should_capitalize_first_letter(&self.0)
            && let Some(i) = msg.find(|c: char| c.is_ascii_alphanumeric())
        {
            msg[i..i + 1].make_ascii_uppercase();
        }
        msg
    }

    fn render_long_inner(&self, format: Format) -> String {
        let on = if self.omit_preposition() {
            " "
        } else if self.should_use_with_in_description() {
            " with "
        } else {
            " on "
        };

        let mut msg = if matches!(format, Format::LongHtml) {
            format!("Available{on}<strong>{}</strong>", Display(&self.0, format))
        } else {
            format!("Available{on}{}", Display(&self.0, format))
        };
        if self.should_append_only_to_description() {
            msg.push_str(" only");
        }
        msg
    }

    /// Renders the configuration for long display, as a long HTML description.
    pub(crate) fn render_long_html(&self) -> String {
        let mut msg = self.render_long_inner(Format::LongHtml);
        msg.push('.');
        msg
    }

    /// Renders the configuration for long display, as a long plain text description.
    pub(crate) fn render_long_plain(&self) -> String {
        self.render_long_inner(Format::LongPlain)
    }

    fn should_append_only_to_description(&self) -> bool {
        match self.0 {
            CfgEntry::Any(..)
            | CfgEntry::All(..)
            | CfgEntry::NameValue { .. }
            | CfgEntry::Version(..)
            | CfgEntry::Not(box CfgEntry::NameValue { .. }, _) => true,
            CfgEntry::Not(..) | CfgEntry::Bool(..) => false,
        }
    }

    fn should_use_with_in_description(&self) -> bool {
        matches!(self.0, CfgEntry::NameValue { name, .. } if name == sym::target_feature)
    }

    /// Attempt to simplify this cfg by assuming that `assume` is already known to be true, will
    /// return `None` if simplification managed to completely eliminate any requirements from this
    /// `Cfg`.
    ///
    /// See `tests::test_simplify_with` for examples.
    pub(crate) fn simplify_with(&self, assume: &Self) -> Option<Self> {
        if self.0.is_equivalent_to(&assume.0) {
            None
        } else if let CfgEntry::All(a, _) = &self.0 {
            let mut sub_cfgs: ThinVec<CfgEntry> = if let CfgEntry::All(b, _) = &assume.0 {
                a.iter().filter(|a| !b.iter().any(|b| a.is_equivalent_to(b))).cloned().collect()
            } else {
                a.iter().filter(|&a| !a.is_equivalent_to(&assume.0)).cloned().collect()
            };
            let len = sub_cfgs.len();
            match len {
                0 => None,
                1 => sub_cfgs.pop().map(Cfg),
                _ => Some(Cfg(CfgEntry::All(sub_cfgs, DUMMY_SP))),
            }
        } else if let CfgEntry::All(b, _) = &assume.0
            && b.iter().any(|b| b.is_equivalent_to(&self.0))
        {
            None
        } else {
            Some(self.clone())
        }
    }

    fn omit_preposition(&self) -> bool {
        matches!(self.0, CfgEntry::Bool(..))
    }

    pub(crate) fn inner(&self) -> &CfgEntry {
        &self.0
    }
}

impl ops::Not for Cfg {
    type Output = Cfg;
    fn not(self) -> Cfg {
        Cfg(match self.0 {
            CfgEntry::Bool(v, s) => CfgEntry::Bool(!v, s),
            CfgEntry::Not(cfg, _) => *cfg,
            s => CfgEntry::Not(Box::new(s), DUMMY_SP),
        })
    }
}

impl ops::BitAndAssign for Cfg {
    fn bitand_assign(&mut self, other: Cfg) {
        match (&mut self.0, other.0) {
            (CfgEntry::Bool(false, _), _) | (_, CfgEntry::Bool(true, _)) => {}
            (s, CfgEntry::Bool(false, _)) => *s = CfgEntry::Bool(false, DUMMY_SP),
            (s @ CfgEntry::Bool(true, _), b) => *s = b,
            (CfgEntry::All(a, _), CfgEntry::All(ref mut b, _)) => {
                for c in b.drain(..) {
                    if !a.iter().any(|a| a.is_equivalent_to(&c)) {
                        a.push(c);
                    }
                }
            }
            (CfgEntry::All(a, _), ref mut b) => {
                if !a.iter().any(|a| a.is_equivalent_to(b)) {
                    a.push(mem::replace(b, CfgEntry::Bool(true, DUMMY_SP)));
                }
            }
            (s, CfgEntry::All(mut a, _)) => {
                let b = mem::replace(s, CfgEntry::Bool(true, DUMMY_SP));
                if !a.iter().any(|a| a.is_equivalent_to(&b)) {
                    a.push(b);
                }
                *s = CfgEntry::All(a, DUMMY_SP);
            }
            (s, b) => {
                if !s.is_equivalent_to(&b) {
                    let a = mem::replace(s, CfgEntry::Bool(true, DUMMY_SP));
                    *s = CfgEntry::All(thin_vec![a, b], DUMMY_SP);
                }
            }
        }
    }
}

impl ops::BitAnd for Cfg {
    type Output = Cfg;
    fn bitand(mut self, other: Cfg) -> Cfg {
        self &= other;
        self
    }
}

impl ops::BitOrAssign for Cfg {
    fn bitor_assign(&mut self, other: Cfg) {
        match (&mut self.0, other.0) {
            (CfgEntry::Bool(true, _), _)
            | (_, CfgEntry::Bool(false, _))
            | (_, CfgEntry::Bool(true, _)) => {}
            (s @ CfgEntry::Bool(false, _), b) => *s = b,
            (CfgEntry::Any(a, _), CfgEntry::Any(ref mut b, _)) => {
                for c in b.drain(..) {
                    if !a.iter().any(|a| a.is_equivalent_to(&c)) {
                        a.push(c);
                    }
                }
            }
            (CfgEntry::Any(a, _), ref mut b) => {
                if !a.iter().any(|a| a.is_equivalent_to(b)) {
                    a.push(mem::replace(b, CfgEntry::Bool(true, DUMMY_SP)));
                }
            }
            (s, CfgEntry::Any(mut a, _)) => {
                let b = mem::replace(s, CfgEntry::Bool(true, DUMMY_SP));
                if !a.iter().any(|a| a.is_equivalent_to(&b)) {
                    a.push(b);
                }
                *s = CfgEntry::Any(a, DUMMY_SP);
            }
            (s, b) => {
                if !s.is_equivalent_to(&b) {
                    let a = mem::replace(s, CfgEntry::Bool(true, DUMMY_SP));
                    *s = CfgEntry::Any(thin_vec![a, b], DUMMY_SP);
                }
            }
        }
    }
}

impl ops::BitOr for Cfg {
    type Output = Cfg;
    fn bitor(mut self, other: Cfg) -> Cfg {
        self |= other;
        self
    }
}

#[derive(Clone, Copy)]
enum Format {
    LongHtml,
    LongPlain,
    ShortHtml,
}

impl Format {
    fn is_long(self) -> bool {
        match self {
            Format::LongHtml | Format::LongPlain => true,
            Format::ShortHtml => false,
        }
    }

    fn is_html(self) -> bool {
        match self {
            Format::LongHtml | Format::ShortHtml => true,
            Format::LongPlain => false,
        }
    }

    fn escape(self, s: &str) -> impl fmt::Display {
        if self.is_html() { Either::Left(Escape(s)) } else { Either::Right(s) }
    }
}

/// Pretty-print wrapper for a `Cfg`. Also indicates what form of rendering should be used.
struct Display<'a>(&'a CfgEntry, Format);

impl Display<'_> {
    fn code_wrappers(&self) -> Wrapped<&'static str> {
        if self.1.is_html() { Wrapped::with("<code>", "</code>") } else { Wrapped::with("`", "`") }
    }

    fn display_sub_cfgs(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        sub_cfgs: &[CfgEntry],
        separator: &str,
    ) -> fmt::Result {
        use fmt::Display as _;

        let short_longhand = self.1.is_long() && {
            let all_crate_features = sub_cfgs.iter().all(|sub_cfg| {
                matches!(sub_cfg, CfgEntry::NameValue { name: sym::feature, value: Some(_), .. })
            });
            let all_target_features = sub_cfgs.iter().all(|sub_cfg| {
                matches!(
                    sub_cfg,
                    CfgEntry::NameValue { name: sym::target_feature, value: Some(_), .. }
                )
            });

            if all_crate_features {
                fmt.write_str("crate features ")?;
                true
            } else if all_target_features {
                fmt.write_str("target features ")?;
                true
            } else {
                false
            }
        };

        fmt::from_fn(|f| {
            sub_cfgs
                .iter()
                .map(|sub_cfg| {
                    if let CfgEntry::NameValue { value: Some(feat), .. } = sub_cfg
                        && short_longhand
                    {
                        Either::Left(self.code_wrappers().wrap(feat))
                    } else {
                        Either::Right(
                            Wrapped::with_parens()
                                .when(!is_all_cfg(sub_cfg))
                                .wrap(Display(sub_cfg, self.1)),
                        )
                    }
                })
                .joined(separator, f)
        })
        .fmt(fmt)?;

        Ok(())
    }
}

impl fmt::Display for Display<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            CfgEntry::Not(box CfgEntry::Any(sub_cfgs, _), _) => {
                let separator = if sub_cfgs.iter().all(is_simple_cfg) { " nor " } else { ", nor " };
                fmt.write_str("neither ")?;

                sub_cfgs
                    .iter()
                    .map(|sub_cfg| {
                        Wrapped::with_parens()
                            .when(!is_all_cfg(sub_cfg))
                            .wrap(Display(sub_cfg, self.1))
                    })
                    .joined(separator, fmt)
            }
            CfgEntry::Not(box simple @ CfgEntry::NameValue { .. }, _) => {
                write!(fmt, "non-{}", Display(simple, self.1))
            }
            CfgEntry::Not(box c, _) => write!(fmt, "not ({})", Display(c, self.1)),

            CfgEntry::Any(sub_cfgs, _) => {
                let separator = if sub_cfgs.iter().all(is_simple_cfg) { " or " } else { ", or " };
                self.display_sub_cfgs(fmt, sub_cfgs.as_slice(), separator)
            }
            CfgEntry::All(sub_cfgs, _) => self.display_sub_cfgs(fmt, sub_cfgs.as_slice(), " and "),

            CfgEntry::Bool(v, _) => {
                if *v {
                    fmt.write_str("everywhere")
                } else {
                    fmt.write_str("nowhere")
                }
            }

            &CfgEntry::NameValue { name, value, .. } => {
                let human_readable = match (*name, value) {
                    (sym::unix, None) => "Unix",
                    (sym::windows, None) => "Windows",
                    (sym::debug_assertions, None) => "debug-assertions enabled",
                    (sym::target_os, Some(os)) => match os.as_str() {
                        "android" => "Android",
                        "cygwin" => "Cygwin",
                        "dragonfly" => "DragonFly BSD",
                        "emscripten" => "Emscripten",
                        "freebsd" => "FreeBSD",
                        "fuchsia" => "Fuchsia",
                        "haiku" => "Haiku",
                        "hermit" => "HermitCore",
                        "illumos" => "illumos",
                        "ios" => "iOS",
                        "l4re" => "L4Re",
                        "linux" => "Linux",
                        "macos" => "macOS",
                        "netbsd" => "NetBSD",
                        "openbsd" => "OpenBSD",
                        "redox" => "Redox",
                        "solaris" => "Solaris",
                        "tvos" => "tvOS",
                        "wasi" => "WASI",
                        "watchos" => "watchOS",
                        "windows" => "Windows",
                        "visionos" => "visionOS",
                        _ => "",
                    },
                    (sym::target_arch, Some(arch)) => match arch.as_str() {
                        "aarch64" => "AArch64",
                        "arm" => "ARM",
                        "loongarch32" => "LoongArch LA32",
                        "loongarch64" => "LoongArch LA64",
                        "m68k" => "M68k",
                        "csky" => "CSKY",
                        "mips" => "MIPS",
                        "mips32r6" => "MIPS Release 6",
                        "mips64" => "MIPS-64",
                        "mips64r6" => "MIPS-64 Release 6",
                        "msp430" => "MSP430",
                        "powerpc" => "PowerPC",
                        "powerpc64" => "PowerPC-64",
                        "riscv32" => "RISC-V RV32",
                        "riscv64" => "RISC-V RV64",
                        "s390x" => "s390x",
                        "sparc64" => "SPARC64",
                        "wasm32" | "wasm64" => "WebAssembly",
                        "x86" => "x86",
                        "x86_64" => "x86-64",
                        _ => "",
                    },
                    (sym::target_vendor, Some(vendor)) => match vendor.as_str() {
                        "apple" => "Apple",
                        "pc" => "PC",
                        "sun" => "Sun",
                        "fortanix" => "Fortanix",
                        _ => "",
                    },
                    (sym::target_env, Some(env)) => match env.as_str() {
                        "gnu" => "GNU",
                        "msvc" => "MSVC",
                        "musl" => "musl",
                        "newlib" => "Newlib",
                        "uclibc" => "uClibc",
                        "sgx" => "SGX",
                        _ => "",
                    },
                    (sym::target_endian, Some(endian)) => {
                        return write!(fmt, "{endian}-endian");
                    }
                    (sym::target_pointer_width, Some(bits)) => {
                        return write!(fmt, "{bits}-bit");
                    }
                    (sym::target_feature, Some(feat)) => match self.1 {
                        Format::LongHtml => {
                            return write!(fmt, "target feature <code>{feat}</code>");
                        }
                        Format::LongPlain => return write!(fmt, "target feature `{feat}`"),
                        Format::ShortHtml => return write!(fmt, "<code>{feat}</code>"),
                    },
                    (sym::feature, Some(feat)) => match self.1 {
                        Format::LongHtml => {
                            return write!(fmt, "crate feature <code>{feat}</code>");
                        }
                        Format::LongPlain => return write!(fmt, "crate feature `{feat}`"),
                        Format::ShortHtml => return write!(fmt, "<code>{feat}</code>"),
                    },
                    _ => "",
                };
                if !human_readable.is_empty() {
                    fmt.write_str(human_readable)
                } else {
                    let value = value
                        .map(|v| fmt::from_fn(move |f| write!(f, "={}", self.1.escape(v.as_str()))))
                        .maybe_display();
                    self.code_wrappers()
                        .wrap(format_args!("{}{value}", self.1.escape(name.as_str())))
                        .fmt(fmt)
                }
            }

            CfgEntry::Version(..) => {
                // FIXME: Should we handle it?
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct NameValueCfg {
    name: Symbol,
    value: Option<Symbol>,
}

impl NameValueCfg {
    fn new(name: Symbol) -> Self {
        Self { name, value: None }
    }
}

impl<'a> From<&'a CfgEntry> for NameValueCfg {
    fn from(cfg: &'a CfgEntry) -> Self {
        match cfg {
            CfgEntry::NameValue { name, value, .. } => NameValueCfg { name: *name, value: *value },
            _ => NameValueCfg { name: sym::empty, value: None },
        }
    }
}

impl<'a> From<&'a attrs::CfgInfo> for NameValueCfg {
    fn from(cfg: &'a attrs::CfgInfo) -> Self {
        Self { name: cfg.name, value: cfg.value.map(|(value, _)| value) }
    }
}

/// This type keeps track of (doc) cfg information as we go down the item tree.
#[derive(Clone, Debug)]
pub(crate) struct CfgInfo {
    /// List of currently active `doc(auto_cfg(hide(...)))` cfgs, minus currently active
    /// `doc(auto_cfg(show(...)))` cfgs.
    hidden_cfg: FxHashSet<NameValueCfg>,
    /// Current computed `cfg`. Each time we enter a new item, this field is updated as well while
    /// taking into account the `hidden_cfg` information.
    current_cfg: Cfg,
    /// Whether the `doc(auto_cfg())` feature is enabled or not at this point.
    auto_cfg_active: bool,
    /// If the parent item used `doc(cfg(...))`, then we don't want to overwrite `current_cfg`,
    /// instead we will concatenate with it. However, if it's not the case, we need to overwrite
    /// `current_cfg`.
    parent_is_doc_cfg: bool,
}

impl Default for CfgInfo {
    fn default() -> Self {
        Self {
            hidden_cfg: FxHashSet::from_iter([
                NameValueCfg::new(sym::test),
                NameValueCfg::new(sym::doc),
                NameValueCfg::new(sym::doctest),
            ]),
            current_cfg: Cfg(CfgEntry::Bool(true, DUMMY_SP)),
            auto_cfg_active: true,
            parent_is_doc_cfg: false,
        }
    }
}

fn show_hide_show_conflict_error(
    tcx: TyCtxt<'_>,
    item_span: rustc_span::Span,
    previous: rustc_span::Span,
) {
    let mut diag = tcx.sess.dcx().struct_span_err(
        item_span,
        format!(
            "same `cfg` was in `auto_cfg(hide(...))` and `auto_cfg(show(...))` on the same item"
        ),
    );
    diag.span_note(previous, "first change was here");
    diag.emit();
}

/// This functions updates the `hidden_cfg` field of the provided `cfg_info` argument.
///
/// It also checks if a same `cfg` is present in both `auto_cfg(hide(...))` and
/// `auto_cfg(show(...))` on the same item and emits an error if it's the case.
///
/// Because we go through a list of `cfg`s, we keep track of the `cfg`s we saw in `new_show_attrs`
/// and in `new_hide_attrs` arguments.
fn handle_auto_cfg_hide_show(
    tcx: TyCtxt<'_>,
    cfg_info: &mut CfgInfo,
    attr: &CfgHideShow,
    new_show_attrs: &mut FxHashMap<(Symbol, Option<Symbol>), rustc_span::Span>,
    new_hide_attrs: &mut FxHashMap<(Symbol, Option<Symbol>), rustc_span::Span>,
) {
    for value in &attr.values {
        let simple = NameValueCfg::from(value);
        if attr.kind == HideOrShow::Show {
            if let Some(span) = new_hide_attrs.get(&(simple.name, simple.value)) {
                show_hide_show_conflict_error(tcx, value.span_for_name_and_value(), *span);
            } else {
                new_show_attrs.insert((simple.name, simple.value), value.span_for_name_and_value());
            }
            cfg_info.hidden_cfg.remove(&simple);
        } else {
            if let Some(span) = new_show_attrs.get(&(simple.name, simple.value)) {
                show_hide_show_conflict_error(tcx, value.span_for_name_and_value(), *span);
            } else {
                new_hide_attrs.insert((simple.name, simple.value), value.span_for_name_and_value());
            }
            cfg_info.hidden_cfg.insert(simple);
        }
    }
}

pub(crate) fn extract_cfg_from_attrs<'a, I: Iterator<Item = &'a hir::Attribute> + Clone>(
    attrs: I,
    tcx: TyCtxt<'_>,
    cfg_info: &mut CfgInfo,
) -> Option<Arc<Cfg>> {
    fn check_changed_auto_active_status(
        changed_auto_active_status: &mut Option<rustc_span::Span>,
        attr_span: Span,
        cfg_info: &mut CfgInfo,
        tcx: TyCtxt<'_>,
        new_value: bool,
    ) -> bool {
        if let Some(first_change) = changed_auto_active_status {
            if cfg_info.auto_cfg_active != new_value {
                tcx.sess
                    .dcx()
                    .struct_span_err(
                        vec![*first_change, attr_span],
                        "`auto_cfg` was disabled and enabled more than once on the same item",
                    )
                    .emit();
                return true;
            }
        } else {
            *changed_auto_active_status = Some(attr_span);
        }
        cfg_info.auto_cfg_active = new_value;
        false
    }

    let mut new_show_attrs = FxHashMap::default();
    let mut new_hide_attrs = FxHashMap::default();

    let mut doc_cfg = attrs
        .clone()
        .filter_map(|attr| match attr {
            Attribute::Parsed(AttributeKind::Doc(d)) if !d.cfg.is_empty() => Some(d),
            _ => None,
        })
        .peekable();
    // If the item uses `doc(cfg(...))`, then we ignore the other `cfg(...)` attributes.
    if doc_cfg.peek().is_some() {
        // We overwrite existing `cfg`.
        if !cfg_info.parent_is_doc_cfg {
            cfg_info.current_cfg = Cfg(CfgEntry::Bool(true, DUMMY_SP));
            cfg_info.parent_is_doc_cfg = true;
        }
        for attr in doc_cfg {
            for new_cfg in attr.cfg.clone() {
                cfg_info.current_cfg &= Cfg(new_cfg);
            }
        }
    } else {
        cfg_info.parent_is_doc_cfg = false;
    }

    let mut changed_auto_active_status = None;

    // We get all `doc(auto_cfg)`, `cfg` and `target_feature` attributes.
    for attr in attrs {
        if let Attribute::Parsed(AttributeKind::Doc(d)) = attr {
            for (new_value, span) in &d.auto_cfg_change {
                if check_changed_auto_active_status(
                    &mut changed_auto_active_status,
                    *span,
                    cfg_info,
                    tcx,
                    *new_value,
                ) {
                    return None;
                }
            }
            if let Some((_, span)) = d.auto_cfg.first() {
                if check_changed_auto_active_status(
                    &mut changed_auto_active_status,
                    *span,
                    cfg_info,
                    tcx,
                    true,
                ) {
                    return None;
                }
                for (value, _) in &d.auto_cfg {
                    handle_auto_cfg_hide_show(
                        tcx,
                        cfg_info,
                        value,
                        &mut new_show_attrs,
                        &mut new_hide_attrs,
                    );
                }
            }
        } else if let hir::Attribute::Parsed(AttributeKind::TargetFeature { features, .. }) = attr {
            // Treat `#[target_feature(enable = "feat")]` attributes as if they were
            // `#[doc(cfg(target_feature = "feat"))]` attributes as well.
            for (feature, _) in features {
                cfg_info.current_cfg &= Cfg(CfgEntry::NameValue {
                    name: sym::target_feature,
                    value: Some(*feature),
                    span: DUMMY_SP,
                });
            }
            continue;
        } else if !cfg_info.parent_is_doc_cfg
            && let hir::Attribute::Parsed(AttributeKind::CfgTrace(cfgs)) = attr
        {
            for (new_cfg, _) in cfgs {
                cfg_info.current_cfg &= Cfg(new_cfg.clone());
            }
        }
    }

    // If `doc(auto_cfg)` feature is disabled and `doc(cfg())` wasn't used, there is nothing
    // to be done here.
    if !cfg_info.auto_cfg_active && !cfg_info.parent_is_doc_cfg {
        None
    } else if cfg_info.parent_is_doc_cfg {
        if matches!(cfg_info.current_cfg.0, CfgEntry::Bool(true, _)) {
            None
        } else {
            Some(Arc::new(cfg_info.current_cfg.clone()))
        }
    } else {
        // If `doc(auto_cfg)` feature is enabled, we want to collect all `cfg` items, we remove the
        // hidden ones afterward.
        match strip_hidden(&cfg_info.current_cfg.0, &cfg_info.hidden_cfg) {
            None | Some(CfgEntry::Bool(true, _)) => None,
            Some(cfg) => Some(Arc::new(Cfg(cfg))),
        }
    }
}
