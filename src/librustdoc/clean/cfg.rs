//! The representation of a `#[doc(cfg(...))]` attribute.

// FIXME: Once the portability lint RFC is implemented (see tracking issue #41619),
// switch to use those structures instead.

use std::fmt::{self, Write};
use std::mem;
use std::ops;

use rustc_ast::{LitKind, MetaItem, MetaItemKind, NestedMetaItem};
use rustc_feature::Features;
use rustc_session::parse::ParseSess;
use rustc_span::symbol::{sym, Symbol};

use rustc_span::Span;

use crate::html::escape::Escape;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
crate enum Cfg {
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
crate struct InvalidCfgError {
    crate msg: &'static str,
    crate span: Span,
}

impl Cfg {
    /// Parses a `NestedMetaItem` into a `Cfg`.
    fn parse_nested(nested_cfg: &NestedMetaItem) -> Result<Cfg, InvalidCfgError> {
        match nested_cfg {
            NestedMetaItem::MetaItem(ref cfg) => Cfg::parse(cfg),
            NestedMetaItem::Literal(ref lit) => {
                Err(InvalidCfgError { msg: "unexpected literal", span: lit.span })
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
    crate fn parse(cfg: &MetaItem) -> Result<Cfg, InvalidCfgError> {
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
            MetaItemKind::Word => Ok(Cfg::Cfg(name, None)),
            MetaItemKind::NameValue(ref lit) => match lit.kind {
                LitKind::Str(value, _) => Ok(Cfg::Cfg(name, Some(value))),
                _ => Err(InvalidCfgError {
                    // FIXME: if the main #[cfg] syntax decided to support non-string literals,
                    // this should be changed as well.
                    msg: "value of cfg option should be a string literal",
                    span: lit.span,
                }),
            },
            MetaItemKind::List(ref items) => {
                let mut sub_cfgs = items.iter().map(Cfg::parse_nested);
                match name {
                    sym::all => sub_cfgs.fold(Ok(Cfg::True), |x, y| Ok(x? & y?)),
                    sym::any => sub_cfgs.fold(Ok(Cfg::False), |x, y| Ok(x? | y?)),
                    sym::not => {
                        if sub_cfgs.len() == 1 {
                            Ok(!sub_cfgs.next().unwrap()?)
                        } else {
                            Err(InvalidCfgError { msg: "expected 1 cfg-pattern", span: cfg.span })
                        }
                    }
                    _ => Err(InvalidCfgError { msg: "invalid predicate", span: cfg.span }),
                }
            }
        }
    }

    /// Checks whether the given configuration can be matched in the current session.
    ///
    /// Equivalent to `attr::cfg_matches`.
    // FIXME: Actually make use of `features`.
    crate fn matches(&self, parse_sess: &ParseSess, features: Option<&Features>) -> bool {
        match *self {
            Cfg::False => false,
            Cfg::True => true,
            Cfg::Not(ref child) => !child.matches(parse_sess, features),
            Cfg::All(ref sub_cfgs) => {
                sub_cfgs.iter().all(|sub_cfg| sub_cfg.matches(parse_sess, features))
            }
            Cfg::Any(ref sub_cfgs) => {
                sub_cfgs.iter().any(|sub_cfg| sub_cfg.matches(parse_sess, features))
            }
            Cfg::Cfg(name, value) => parse_sess.config.contains(&(name, value)),
        }
    }

    /// Whether the configuration consists of just `Cfg` or `Not`.
    fn is_simple(&self) -> bool {
        match *self {
            Cfg::False | Cfg::True | Cfg::Cfg(..) | Cfg::Not(..) => true,
            Cfg::All(..) | Cfg::Any(..) => false,
        }
    }

    /// Whether the configuration consists of just `Cfg`, `Not` or `All`.
    fn is_all(&self) -> bool {
        match *self {
            Cfg::False | Cfg::True | Cfg::Cfg(..) | Cfg::Not(..) | Cfg::All(..) => true,
            Cfg::Any(..) => false,
        }
    }

    /// Renders the configuration for human display, as a short HTML description.
    pub(crate) fn render_short_html(&self) -> String {
        let mut msg = Display(self, Format::ShortHtml).to_string();
        if self.should_capitalize_first_letter() {
            if let Some(i) = msg.find(|c: char| c.is_ascii_alphanumeric()) {
                msg[i..i + 1].make_ascii_uppercase();
            }
        }
        msg
    }

    /// Renders the configuration for long display, as a long HTML description.
    pub(crate) fn render_long_html(&self) -> String {
        let on = if self.should_use_with_in_description() { "with" } else { "on" };

        let mut msg = format!(
            "This is supported {} <strong>{}</strong>",
            on,
            Display(self, Format::LongHtml)
        );
        if self.should_append_only_to_description() {
            msg.push_str(" only");
        }
        msg.push('.');
        msg
    }

    /// Renders the configuration for long display, as a long plain text description.
    pub(crate) fn render_long_plain(&self) -> String {
        let on = if self.should_use_with_in_description() { "with" } else { "on" };

        let mut msg = format!("This is supported {} {}", on, Display(self, Format::LongPlain));
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
        match *self {
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
    pub(crate) fn simplify_with(&self, assume: &Cfg) -> Option<Cfg> {
        if self == assume {
            return None;
        }

        if let Cfg::All(a) = self {
            let mut sub_cfgs: Vec<Cfg> = if let Cfg::All(b) = assume {
                a.iter().filter(|a| !b.contains(a)).cloned().collect()
            } else {
                a.iter().filter(|&a| a != assume).cloned().collect()
            };
            let len = sub_cfgs.len();
            return match len {
                0 => None,
                1 => sub_cfgs.pop(),
                _ => Some(Cfg::All(sub_cfgs)),
            };
        } else if let Cfg::All(b) = assume {
            if b.contains(self) {
                return None;
            }
        }

        Some(self.clone())
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
            (&mut Cfg::False, _) | (_, Cfg::True) => {}
            (s, Cfg::False) => *s = Cfg::False,
            (s @ &mut Cfg::True, b) => *s = b,
            (&mut Cfg::All(ref mut a), Cfg::All(ref mut b)) => {
                for c in b.drain(..) {
                    if !a.contains(&c) {
                        a.push(c);
                    }
                }
            }
            (&mut Cfg::All(ref mut a), ref mut b) => {
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
            (&mut Cfg::True, _) | (_, Cfg::False) => {}
            (s, Cfg::True) => *s = Cfg::True,
            (s @ &mut Cfg::False, b) => *s = b,
            (&mut Cfg::Any(ref mut a), Cfg::Any(ref mut b)) => {
                for c in b.drain(..) {
                    if !a.contains(&c) {
                        a.push(c);
                    }
                }
            }
            (&mut Cfg::Any(ref mut a), ref mut b) => {
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

impl<'a> fmt::Display for Display<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self.0 {
            Cfg::Not(ref child) => match **child {
                Cfg::Any(ref sub_cfgs) => {
                    let separator =
                        if sub_cfgs.iter().all(Cfg::is_simple) { " nor " } else { ", nor " };
                    for (i, sub_cfg) in sub_cfgs.iter().enumerate() {
                        fmt.write_str(if i == 0 { "neither " } else { separator })?;
                        write_with_opt_paren(fmt, !sub_cfg.is_all(), Display(sub_cfg, self.1))?;
                    }
                    Ok(())
                }
                ref simple @ Cfg::Cfg(..) => write!(fmt, "non-{}", Display(simple, self.1)),
                ref c => write!(fmt, "not ({})", Display(c, self.1)),
            },

            Cfg::Any(ref sub_cfgs) => {
                let separator = if sub_cfgs.iter().all(Cfg::is_simple) { " or " } else { ", or " };

                let short_longhand = self.1.is_long() && {
                    let all_crate_features = sub_cfgs
                        .iter()
                        .all(|sub_cfg| matches!(sub_cfg, Cfg::Cfg(sym::feature, Some(_))));
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

                for (i, sub_cfg) in sub_cfgs.iter().enumerate() {
                    if i != 0 {
                        fmt.write_str(separator)?;
                    }
                    if let (true, Cfg::Cfg(_, Some(feat))) = (short_longhand, sub_cfg) {
                        if self.1.is_html() {
                            write!(fmt, "<code>{}</code>", feat)?;
                        } else {
                            write!(fmt, "`{}`", feat)?;
                        }
                    } else {
                        write_with_opt_paren(fmt, !sub_cfg.is_all(), Display(sub_cfg, self.1))?;
                    }
                }
                Ok(())
            }

            Cfg::All(ref sub_cfgs) => {
                let short_longhand = self.1.is_long() && {
                    let all_crate_features = sub_cfgs
                        .iter()
                        .all(|sub_cfg| matches!(sub_cfg, Cfg::Cfg(sym::feature, Some(_))));
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

                for (i, sub_cfg) in sub_cfgs.iter().enumerate() {
                    if i != 0 {
                        fmt.write_str(" and ")?;
                    }
                    if let (true, Cfg::Cfg(_, Some(feat))) = (short_longhand, sub_cfg) {
                        if self.1.is_html() {
                            write!(fmt, "<code>{}</code>", feat)?;
                        } else {
                            write!(fmt, "`{}`", feat)?;
                        }
                    } else {
                        write_with_opt_paren(fmt, !sub_cfg.is_simple(), Display(sub_cfg, self.1))?;
                    }
                }
                Ok(())
            }

            Cfg::True => fmt.write_str("everywhere"),
            Cfg::False => fmt.write_str("nowhere"),

            Cfg::Cfg(name, value) => {
                let human_readable = match (name, value) {
                    (sym::unix, None) => "Unix",
                    (sym::windows, None) => "Windows",
                    (sym::debug_assertions, None) => "debug-assertions enabled",
                    (sym::target_os, Some(os)) => match &*os.as_str() {
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
                        "windows" => "Windows",
                        _ => "",
                    },
                    (sym::target_arch, Some(arch)) => match &*arch.as_str() {
                        "aarch64" => "AArch64",
                        "arm" => "ARM",
                        "asmjs" => "JavaScript",
                        "mips" => "MIPS",
                        "mips64" => "MIPS-64",
                        "msp430" => "MSP430",
                        "powerpc" => "PowerPC",
                        "powerpc64" => "PowerPC-64",
                        "s390x" => "s390x",
                        "sparc64" => "SPARC64",
                        "wasm32" => "WebAssembly",
                        "x86" => "x86",
                        "x86_64" => "x86-64",
                        _ => "",
                    },
                    (sym::target_vendor, Some(vendor)) => match &*vendor.as_str() {
                        "apple" => "Apple",
                        "pc" => "PC",
                        "rumprun" => "Rumprun",
                        "sun" => "Sun",
                        "fortanix" => "Fortanix",
                        _ => "",
                    },
                    (sym::target_env, Some(env)) => match &*env.as_str() {
                        "gnu" => "GNU",
                        "msvc" => "MSVC",
                        "musl" => "musl",
                        "newlib" => "Newlib",
                        "uclibc" => "uClibc",
                        "sgx" => "SGX",
                        _ => "",
                    },
                    (sym::target_endian, Some(endian)) => return write!(fmt, "{}-endian", endian),
                    (sym::target_pointer_width, Some(bits)) => return write!(fmt, "{}-bit", bits),
                    (sym::target_feature, Some(feat)) => match self.1 {
                        Format::LongHtml => {
                            return write!(fmt, "target feature <code>{}</code>", feat);
                        }
                        Format::LongPlain => return write!(fmt, "target feature `{}`", feat),
                        Format::ShortHtml => return write!(fmt, "<code>{}</code>", feat),
                    },
                    (sym::feature, Some(feat)) => match self.1 {
                        Format::LongHtml => {
                            return write!(fmt, "crate feature <code>{}</code>", feat);
                        }
                        Format::LongPlain => return write!(fmt, "crate feature `{}`", feat),
                        Format::ShortHtml => return write!(fmt, "<code>{}</code>", feat),
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
                            Escape(&name.as_str()),
                            Escape(&v.as_str())
                        )
                    } else {
                        write!(fmt, r#"`{}="{}"`"#, name, v)
                    }
                } else if self.1.is_html() {
                    write!(fmt, "<code>{}</code>", Escape(&name.as_str()))
                } else {
                    write!(fmt, "`{}`", name)
                }
            }
        }
    }
}
