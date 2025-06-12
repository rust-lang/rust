//! The representation of a `#[doc(cfg(...))]` attribute.

// FIXME: Once the portability lint RFC is implemented (see tracking issue #41619),
// switch to use those structures instead.

use std::fmt::{self, Write};
use std::{mem, ops};

use rustc_ast::{LitKind, MetaItem, MetaItemInner, MetaItemKind, MetaItemLit};
use rustc_data_structures::fx::FxHashSet;
use rustc_session::parse::ParseSess;
use rustc_span::Span;
use rustc_span::symbol::{Symbol, sym};

use crate::display::Joined as _;
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

    /// Renders the configuration for long display, as a long HTML description.
    pub(crate) fn render_long_html(&self) -> String {
        let on = if self.omit_preposition() {
            ""
        } else if self.should_use_with_in_description() {
            "with "
        } else {
            "on "
        };

        let mut msg = format!("Available {on}<strong>{}</strong>", Display(self, Format::LongHtml));
        if self.should_append_only_to_description() {
            msg.push_str(" only");
        }
        msg.push('.');
        msg
    }

    /// Renders the configuration for long display, as a long plain text description.
    pub(crate) fn render_long_plain(&self) -> String {
        let on = if self.should_use_with_in_description() { "with" } else { "on" };

        let mut msg = format!("Available {on} {}", Display(self, Format::LongPlain));
        if self.should_append_only_to_description() {
            msg.push_str(" only");
        }
        msg
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
}

/// Pretty-print wrapper for a `Cfg`. Also indicates what form of rendering should be used.
struct Display<'a>(&'a Cfg, Format);

fn write_with_opt_paren<T: fmt::Display>(
    fmt: &mut fmt::Formatter<'_>,
    has_paren: bool,
    obj: T,
) -> fmt::Result {
    if has_paren {
        fmt.write_char('(')?;
    }
    obj.fmt(fmt)?;
    if has_paren {
        fmt.write_char(')')?;
    }
    Ok(())
}

impl Display<'_> {
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
                    fmt::from_fn(move |fmt| {
                        if let Cfg::Cfg(_, Some(feat)) = sub_cfg
                            && short_longhand
                        {
                            if self.1.is_html() {
                                write!(fmt, "<code>{feat}</code>")?;
                            } else {
                                write!(fmt, "`{feat}`")?;
                            }
                        } else {
                            write_with_opt_paren(fmt, !sub_cfg.is_all(), Display(sub_cfg, self.1))?;
                        }
                        Ok(())
                    })
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
                        fmt::from_fn(|fmt| {
                            write_with_opt_paren(fmt, !sub_cfg.is_all(), Display(sub_cfg, self.1))
                        })
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
                } else if let Some(v) = value {
                    if self.1.is_html() {
                        write!(
                            fmt,
                            r#"<code>{}="{}"</code>"#,
                            Escape(name.as_str()),
                            Escape(v.as_str())
                        )
                    } else {
                        write!(fmt, r#"`{name}="{v}"`"#)
                    }
                } else if self.1.is_html() {
                    write!(fmt, "<code>{}</code>", Escape(name.as_str()))
                } else {
                    write!(fmt, "`{name}`")
                }
            }
        }
    }
}
