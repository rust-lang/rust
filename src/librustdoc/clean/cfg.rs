//! The representation of a `#[doc(cfg(...))]` attribute.

// FIXME: Once the portability lint RFC is implemented (see tracking issue #41619),
// switch to use those structures instead.

use std::sync::Arc;
use std::{fmt, mem, ops};

use itertools::Either;
use rustc_ast::{LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_hir::attrs::AttributeKind;
use rustc_middle::ty::TyCtxt;
use rustc_session::parse::ParseSess;
use rustc_span::Span;
use rustc_span::symbol::{Symbol, sym};
use {rustc_ast as ast, rustc_hir as hir};

use crate::display::{Joined as _, MaybeDisplay, Wrapped};
use crate::html::escape::Escape;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Cfg {
    /// Accepts all configurations.
    True,
    /// Denies all configurations.
    False,
    /// A generic configuration option, e.g., `test` or `target_os = "linux"`.
    Cfg(Symbol, Option<Symbol>),
    /// Negates a configuration requirement, i.e., `not(x)`.
    Not(Box<Cfg>),
    /// Union of a list of configuration requirements, i.e., `any(...)`.
    Any(Vec<Cfg>),
    /// Intersection of a list of configuration requirements, i.e., `all(...)`.
    All(Vec<Cfg>),
}

#[derive(PartialEq, Debug)]
pub(crate) struct InvalidCfgError {
    pub(crate) msg: &'static str,
    pub(crate) span: Span,
}

impl Cfg {
    /// Parses a `MetaItemInner` into a `Cfg`.
    fn parse_nested(
        nested_cfg: &MetaItemInner,
        exclude: &FxHashSet<Cfg>,
    ) -> Result<Option<Cfg>, InvalidCfgError> {
        match nested_cfg {
            MetaItemInner::MetaItem(cfg) => Cfg::parse_without(cfg, exclude),
            MetaItemInner::Lit(MetaItemLit { kind: LitKind::Bool(b), .. }) => match *b {
                true => Ok(Some(Cfg::True)),
                false => Ok(Some(Cfg::False)),
            },
            MetaItemInner::Lit(lit) => {
                Err(InvalidCfgError { msg: "unexpected literal", span: lit.span })
            }
        }
    }

    pub(crate) fn parse_without(
        cfg: &MetaItem,
        exclude: &FxHashSet<Cfg>,
    ) -> Result<Option<Cfg>, InvalidCfgError> {
        let name = match cfg.ident() {
            Some(ident) => ident.name,
            None => {
                return Err(InvalidCfgError {
                    msg: "expected a single identifier",
                    span: cfg.span,
                });
            }
        };
        match cfg.kind {
            MetaItemKind::Word => {
                let cfg = Cfg::Cfg(name, None);
                if exclude.contains(&cfg) { Ok(None) } else { Ok(Some(cfg)) }
            }
            MetaItemKind::NameValue(ref lit) => match lit.kind {
                LitKind::Str(value, _) => {
                    let cfg = Cfg::Cfg(name, Some(value));
                    if exclude.contains(&cfg) { Ok(None) } else { Ok(Some(cfg)) }
                }
                _ => Err(InvalidCfgError {
                    // FIXME: if the main #[cfg] syntax decided to support non-string literals,
                    // this should be changed as well.
                    msg: "value of cfg option should be a string literal",
                    span: lit.span,
                }),
            },
            MetaItemKind::List(ref items) => {
                let orig_len = items.len();
                let mut sub_cfgs =
                    items.iter().filter_map(|i| Cfg::parse_nested(i, exclude).transpose());
                let ret = match name {
                    sym::all => sub_cfgs.try_fold(Cfg::True, |x, y| Ok(x & y?)),
                    sym::any => sub_cfgs.try_fold(Cfg::False, |x, y| Ok(x | y?)),
                    sym::not => {
                        if orig_len == 1 {
                            let mut sub_cfgs = sub_cfgs.collect::<Vec<_>>();
                            if sub_cfgs.len() == 1 {
                                Ok(!sub_cfgs.pop().unwrap()?)
                            } else {
                                return Ok(None);
                            }
                        } else {
                            Err(InvalidCfgError { msg: "expected 1 cfg-pattern", span: cfg.span })
                        }
                    }
                    _ => Err(InvalidCfgError { msg: "invalid predicate", span: cfg.span }),
                };
                match ret {
                    Ok(c) => Ok(Some(c)),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Parses a `MetaItem` into a `Cfg`.
    ///
    /// The `MetaItem` should be the content of the `#[cfg(...)]`, e.g., `unix` or
    /// `target_os = "redox"`.
    ///
    /// If the content is not properly formatted, it will return an error indicating what and where
    /// the error is.
    pub(crate) fn parse(cfg: &MetaItemInner) -> Result<Cfg, InvalidCfgError> {
        Self::parse_nested(cfg, &FxHashSet::default()).map(|ret| ret.unwrap())
    }

    /// Checks whether the given configuration can be matched in the current session.
    ///
    /// Equivalent to `attr::cfg_matches`.
    pub(crate) fn matches(&self, psess: &ParseSess) -> bool {
        match *self {
            Cfg::False => false,
            Cfg::True => true,
            Cfg::Not(ref child) => !child.matches(psess),
            Cfg::All(ref sub_cfgs) => sub_cfgs.iter().all(|sub_cfg| sub_cfg.matches(psess)),
            Cfg::Any(ref sub_cfgs) => sub_cfgs.iter().any(|sub_cfg| sub_cfg.matches(psess)),
            Cfg::Cfg(name, value) => psess.config.contains(&(name, value)),
        }
    }

    /// Whether the configuration consists of just `Cfg` or `Not`.
    fn is_simple(&self) -> bool {
        match self {
            Cfg::False | Cfg::True | Cfg::Cfg(..) | Cfg::Not(..) => true,
            Cfg::All(..) | Cfg::Any(..) => false,
        }
    }

    /// Whether the configuration consists of just `Cfg`, `Not` or `All`.
    fn is_all(&self) -> bool {
        match self {
            Cfg::False | Cfg::True | Cfg::Cfg(..) | Cfg::Not(..) | Cfg::All(..) => true,
            Cfg::Any(..) => false,
        }
    }

    /// Renders the configuration for human display, as a short HTML description.
    pub(crate) fn render_short_html(&self) -> String {
        let mut msg = Display(self, Format::ShortHtml).to_string();
        if self.should_capitalize_first_letter()
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
            format!("Available{on}<strong>{}</strong>", Display(self, format))
        } else {
            format!("Available{on}{}", Display(self, format))
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

    fn should_capitalize_first_letter(&self) -> bool {
        match *self {
            Cfg::False | Cfg::True | Cfg::Not(..) => true,
            Cfg::Any(ref sub_cfgs) | Cfg::All(ref sub_cfgs) => {
                sub_cfgs.first().map(Cfg::should_capitalize_first_letter).unwrap_or(false)
            }
            Cfg::Cfg(name, _) => name == sym::debug_assertions || name == sym::target_endian,
        }
    }

    fn should_append_only_to_description(&self) -> bool {
        match self {
            Cfg::False | Cfg::True => false,
            Cfg::Any(..) | Cfg::All(..) | Cfg::Cfg(..) => true,
            Cfg::Not(box Cfg::Cfg(..)) => true,
            Cfg::Not(..) => false,
        }
    }

    fn should_use_with_in_description(&self) -> bool {
        matches!(self, Cfg::Cfg(sym::target_feature, _))
    }

    /// Attempt to simplify this cfg by assuming that `assume` is already known to be true, will
    /// return `None` if simplification managed to completely eliminate any requirements from this
    /// `Cfg`.
    ///
    /// See `tests::test_simplify_with` for examples.
    pub(crate) fn simplify_with(&self, assume: &Self) -> Option<Self> {
        if self == assume {
            None
        } else if let Cfg::All(a) = self {
            let mut sub_cfgs: Vec<Cfg> = if let Cfg::All(b) = assume {
                a.iter().filter(|a| !b.contains(a)).cloned().collect()
            } else {
                a.iter().filter(|&a| a != assume).cloned().collect()
            };
            let len = sub_cfgs.len();
            match len {
                0 => None,
                1 => sub_cfgs.pop(),
                _ => Some(Cfg::All(sub_cfgs)),
            }
        } else if let Cfg::All(b) = assume
            && b.contains(self)
        {
            None
        } else {
            Some(self.clone())
        }
    }

    fn omit_preposition(&self) -> bool {
        matches!(self, Cfg::True | Cfg::False)
    }

    pub(crate) fn strip_hidden(&self, hidden: &FxHashSet<Cfg>) -> Option<Self> {
        match self {
            Self::True | Self::False => Some(self.clone()),
            Self::Cfg(..) => {
                if !hidden.contains(self) {
                    Some(self.clone())
                } else {
                    None
                }
            }
            Self::Not(cfg) => {
                if let Some(cfg) = cfg.strip_hidden(hidden) {
                    Some(Self::Not(Box::new(cfg)))
                } else {
                    None
                }
            }
            Self::Any(cfgs) => {
                let cfgs =
                    cfgs.iter().filter_map(|cfg| cfg.strip_hidden(hidden)).collect::<Vec<_>>();
                if cfgs.is_empty() { None } else { Some(Self::Any(cfgs)) }
            }
            Self::All(cfgs) => {
                let cfgs =
                    cfgs.iter().filter_map(|cfg| cfg.strip_hidden(hidden)).collect::<Vec<_>>();
                if cfgs.is_empty() { None } else { Some(Self::All(cfgs)) }
            }
        }
    }
}

impl ops::Not for Cfg {
    type Output = Cfg;
    fn not(self) -> Cfg {
        match self {
            Cfg::False => Cfg::True,
            Cfg::True => Cfg::False,
            Cfg::Not(cfg) => *cfg,
            s => Cfg::Not(Box::new(s)),
        }
    }
}

impl ops::BitAndAssign for Cfg {
    fn bitand_assign(&mut self, other: Cfg) {
        match (self, other) {
            (Cfg::False, _) | (_, Cfg::True) => {}
            (s, Cfg::False) => *s = Cfg::False,
            (s @ Cfg::True, b) => *s = b,
            (Cfg::All(a), Cfg::All(ref mut b)) => {
                for c in b.drain(..) {
                    if !a.contains(&c) {
                        a.push(c);
                    }
                }
            }
            (Cfg::All(a), ref mut b) => {
                if !a.contains(b) {
                    a.push(mem::replace(b, Cfg::True));
                }
            }
            (s, Cfg::All(mut a)) => {
                let b = mem::replace(s, Cfg::True);
                if !a.contains(&b) {
                    a.push(b);
                }
                *s = Cfg::All(a);
            }
            (s, b) => {
                if *s != b {
                    let a = mem::replace(s, Cfg::True);
                    *s = Cfg::All(vec![a, b]);
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
        match (self, other) {
            (Cfg::True, _) | (_, Cfg::False) | (_, Cfg::True) => {}
            (s @ Cfg::False, b) => *s = b,
            (Cfg::Any(a), Cfg::Any(ref mut b)) => {
                for c in b.drain(..) {
                    if !a.contains(&c) {
                        a.push(c);
                    }
                }
            }
            (Cfg::Any(a), ref mut b) => {
                if !a.contains(b) {
                    a.push(mem::replace(b, Cfg::True));
                }
            }
            (s, Cfg::Any(mut a)) => {
                let b = mem::replace(s, Cfg::True);
                if !a.contains(&b) {
                    a.push(b);
                }
                *s = Cfg::Any(a);
            }
            (s, b) => {
                if *s != b {
                    let a = mem::replace(s, Cfg::True);
                    *s = Cfg::Any(vec![a, b]);
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
struct Display<'a>(&'a Cfg, Format);

impl Display<'_> {
    fn code_wrappers(&self) -> Wrapped<&'static str> {
        if self.1.is_html() { Wrapped::with("<code>", "</code>") } else { Wrapped::with("`", "`") }
    }

    fn display_sub_cfgs(
        &self,
        fmt: &mut fmt::Formatter<'_>,
        sub_cfgs: &[Cfg],
        separator: &str,
    ) -> fmt::Result {
        use fmt::Display as _;

        let short_longhand = self.1.is_long() && {
            let all_crate_features =
                sub_cfgs.iter().all(|sub_cfg| matches!(sub_cfg, Cfg::Cfg(sym::feature, Some(_))));
            let all_target_features = sub_cfgs
                .iter()
                .all(|sub_cfg| matches!(sub_cfg, Cfg::Cfg(sym::target_feature, Some(_))));

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
                    if let Cfg::Cfg(_, Some(feat)) = sub_cfg
                        && short_longhand
                    {
                        Either::Left(self.code_wrappers().wrap(feat))
                    } else {
                        Either::Right(
                            Wrapped::with_parens()
                                .when(!sub_cfg.is_all())
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
        match self.0 {
            Cfg::Not(box Cfg::Any(sub_cfgs)) => {
                let separator =
                    if sub_cfgs.iter().all(Cfg::is_simple) { " nor " } else { ", nor " };
                fmt.write_str("neither ")?;

                sub_cfgs
                    .iter()
                    .map(|sub_cfg| {
                        Wrapped::with_parens()
                            .when(!sub_cfg.is_all())
                            .wrap(Display(sub_cfg, self.1))
                    })
                    .joined(separator, fmt)
            }
            Cfg::Not(box simple @ Cfg::Cfg(..)) => write!(fmt, "non-{}", Display(simple, self.1)),
            Cfg::Not(box c) => write!(fmt, "not ({})", Display(c, self.1)),

            Cfg::Any(sub_cfgs) => {
                let separator = if sub_cfgs.iter().all(Cfg::is_simple) { " or " } else { ", or " };
                self.display_sub_cfgs(fmt, sub_cfgs, separator)
            }
            Cfg::All(sub_cfgs) => self.display_sub_cfgs(fmt, sub_cfgs, " and "),

            Cfg::True => fmt.write_str("everywhere"),
            Cfg::False => fmt.write_str("nowhere"),

            &Cfg::Cfg(name, value) => {
                let human_readable = match (name, value) {
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
                    (sym::target_endian, Some(endian)) => return write!(fmt, "{endian}-endian"),
                    (sym::target_pointer_width, Some(bits)) => return write!(fmt, "{bits}-bit"),
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
        }
    }
}

/// This type keeps track of (doc) cfg information as we go down the item tree.
#[derive(Clone, Debug)]
pub(crate) struct CfgInfo {
    /// List of currently active `doc(auto_cfg(hide(...)))` cfgs, minus currently active
    /// `doc(auto_cfg(show(...)))` cfgs.
    hidden_cfg: FxHashSet<Cfg>,
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
                Cfg::Cfg(sym::test, None),
                Cfg::Cfg(sym::doc, None),
                Cfg::Cfg(sym::doctest, None),
            ]),
            current_cfg: Cfg::True,
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
    sub_attr: &MetaItemInner,
    is_show: bool,
    new_show_attrs: &mut FxHashMap<(Symbol, Option<Symbol>), rustc_span::Span>,
    new_hide_attrs: &mut FxHashMap<(Symbol, Option<Symbol>), rustc_span::Span>,
) {
    if let MetaItemInner::MetaItem(item) = sub_attr
        && let MetaItemKind::List(items) = &item.kind
    {
        for item in items {
            // FIXME: Report in case `Cfg::parse` reports an error?
            if let Ok(Cfg::Cfg(key, value)) = Cfg::parse(item) {
                if is_show {
                    if let Some(span) = new_hide_attrs.get(&(key, value)) {
                        show_hide_show_conflict_error(tcx, item.span(), *span);
                    } else {
                        new_show_attrs.insert((key, value), item.span());
                    }
                    cfg_info.hidden_cfg.remove(&Cfg::Cfg(key, value));
                } else {
                    if let Some(span) = new_show_attrs.get(&(key, value)) {
                        show_hide_show_conflict_error(tcx, item.span(), *span);
                    } else {
                        new_hide_attrs.insert((key, value), item.span());
                    }
                    cfg_info.hidden_cfg.insert(Cfg::Cfg(key, value));
                }
            }
        }
    }
}

pub(crate) fn extract_cfg_from_attrs<'a, I: Iterator<Item = &'a hir::Attribute> + Clone>(
    attrs: I,
    tcx: TyCtxt<'_>,
    cfg_info: &mut CfgInfo,
) -> Option<Arc<Cfg>> {
    fn single<T: IntoIterator>(it: T) -> Option<T::Item> {
        let mut iter = it.into_iter();
        let item = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        Some(item)
    }

    fn check_changed_auto_active_status(
        changed_auto_active_status: &mut Option<rustc_span::Span>,
        attr: &ast::MetaItem,
        cfg_info: &mut CfgInfo,
        tcx: TyCtxt<'_>,
        new_value: bool,
    ) -> bool {
        if let Some(first_change) = changed_auto_active_status {
            if cfg_info.auto_cfg_active != new_value {
                tcx.sess
                    .dcx()
                    .struct_span_err(
                        vec![*first_change, attr.span],
                        "`auto_cfg` was disabled and enabled more than once on the same item",
                    )
                    .emit();
                return true;
            }
        } else {
            *changed_auto_active_status = Some(attr.span);
        }
        cfg_info.auto_cfg_active = new_value;
        false
    }

    let mut new_show_attrs = FxHashMap::default();
    let mut new_hide_attrs = FxHashMap::default();

    let mut doc_cfg = attrs
        .clone()
        .filter(|attr| attr.has_name(sym::doc))
        .flat_map(|attr| attr.meta_item_list().unwrap_or_default())
        .filter(|attr| attr.has_name(sym::cfg))
        .peekable();
    // If the item uses `doc(cfg(...))`, then we ignore the other `cfg(...)` attributes.
    if doc_cfg.peek().is_some() {
        let sess = tcx.sess;
        // We overwrite existing `cfg`.
        if !cfg_info.parent_is_doc_cfg {
            cfg_info.current_cfg = Cfg::True;
            cfg_info.parent_is_doc_cfg = true;
        }
        for attr in doc_cfg {
            if let Some(cfg_mi) =
                attr.meta_item().and_then(|attr| rustc_expand::config::parse_cfg_old(attr, sess))
            {
                match Cfg::parse(cfg_mi) {
                    Ok(new_cfg) => cfg_info.current_cfg &= new_cfg,
                    Err(e) => {
                        sess.dcx().span_err(e.span, e.msg);
                    }
                }
            }
        }
    } else {
        cfg_info.parent_is_doc_cfg = false;
    }

    let mut changed_auto_active_status = None;

    // We get all `doc(auto_cfg)`, `cfg` and `target_feature` attributes.
    for attr in attrs {
        if let Some(ident) = attr.ident()
            && ident.name == sym::doc
            && let Some(attrs) = attr.meta_item_list()
        {
            for attr in attrs.iter().filter(|attr| attr.has_name(sym::auto_cfg)) {
                let MetaItemInner::MetaItem(attr) = attr else {
                    continue;
                };
                match &attr.kind {
                    MetaItemKind::Word => {
                        if check_changed_auto_active_status(
                            &mut changed_auto_active_status,
                            attr,
                            cfg_info,
                            tcx,
                            true,
                        ) {
                            return None;
                        }
                    }
                    MetaItemKind::NameValue(lit) => {
                        if let LitKind::Bool(value) = lit.kind {
                            if check_changed_auto_active_status(
                                &mut changed_auto_active_status,
                                attr,
                                cfg_info,
                                tcx,
                                value,
                            ) {
                                return None;
                            }
                        }
                    }
                    MetaItemKind::List(sub_attrs) => {
                        if check_changed_auto_active_status(
                            &mut changed_auto_active_status,
                            attr,
                            cfg_info,
                            tcx,
                            true,
                        ) {
                            return None;
                        }
                        for sub_attr in sub_attrs.iter() {
                            if let Some(ident) = sub_attr.ident()
                                && (ident.name == sym::show || ident.name == sym::hide)
                            {
                                handle_auto_cfg_hide_show(
                                    tcx,
                                    cfg_info,
                                    &sub_attr,
                                    ident.name == sym::show,
                                    &mut new_show_attrs,
                                    &mut new_hide_attrs,
                                );
                            }
                        }
                    }
                }
            }
        } else if let hir::Attribute::Parsed(AttributeKind::TargetFeature { features, .. }) = attr {
            // Treat `#[target_feature(enable = "feat")]` attributes as if they were
            // `#[doc(cfg(target_feature = "feat"))]` attributes as well.
            for (feature, _) in features {
                cfg_info.current_cfg &= Cfg::Cfg(sym::target_feature, Some(*feature));
            }
            continue;
        } else if !cfg_info.parent_is_doc_cfg
            && let Some(ident) = attr.ident()
            && matches!(ident.name, sym::cfg | sym::cfg_trace)
            && let Some(attr) = single(attr.meta_item_list()?)
            && let Ok(new_cfg) = Cfg::parse(&attr)
        {
            cfg_info.current_cfg &= new_cfg;
        }
    }

    // If `doc(auto_cfg)` feature is disabled and `doc(cfg())` wasn't used, there is nothing
    // to be done here.
    if !cfg_info.auto_cfg_active && !cfg_info.parent_is_doc_cfg {
        None
    } else if cfg_info.parent_is_doc_cfg {
        if cfg_info.current_cfg == Cfg::True {
            None
        } else {
            Some(Arc::new(cfg_info.current_cfg.clone()))
        }
    } else {
        // If `doc(auto_cfg)` feature is enabled, we want to collect all `cfg` items, we remove the
        // hidden ones afterward.
        match cfg_info.current_cfg.strip_hidden(&cfg_info.hidden_cfg) {
            None | Some(Cfg::True) => None,
            Some(cfg) => Some(Arc::new(cfg)),
        }
    }
}
