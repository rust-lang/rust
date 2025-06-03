use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{ToTokens, TokenStreamExt, quote};
use serde_with::{DeserializeFromStr, SerializeDisplay};
use std::str::pattern::Pattern;
use std::{fmt, str::FromStr};

use crate::context::LocalContext;
use crate::fn_suffix::make_neon_suffix;
use crate::typekinds::{ToRepr, TypeRepr};
use crate::wildcards::Wildcard;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WildStringPart {
    String(String),
    Wildcard(Wildcard),
}

/// Wildcard-able string
#[derive(Debug, Clone, PartialEq, Eq, Default, SerializeDisplay, DeserializeFromStr)]
pub struct WildString(pub Vec<WildStringPart>);

impl WildString {
    pub fn has_wildcards(&self) -> bool {
        for part in self.0.iter() {
            if let WildStringPart::Wildcard(..) = part {
                return true;
            }
        }

        false
    }

    pub fn wildcards(&self) -> impl Iterator<Item = &Wildcard> + '_ {
        self.0.iter().filter_map(|part| match part {
            WildStringPart::Wildcard(w) => Some(w),
            _ => None,
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &WildStringPart> + '_ {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut WildStringPart> + '_ {
        self.0.iter_mut()
    }

    pub fn starts_with(&self, s2: &str) -> bool {
        self.to_string().starts_with(s2)
    }

    pub fn prepend_str(&mut self, s: impl Into<String>) {
        self.0.insert(0, WildStringPart::String(s.into()))
    }

    pub fn push_str(&mut self, s: impl Into<String>) {
        self.0.push(WildStringPart::String(s.into()))
    }

    pub fn push_wildcard(&mut self, w: Wildcard) {
        self.0.push(WildStringPart::Wildcard(w))
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn replace<'a, P>(&'a self, from: P, to: &str) -> WildString
    where
        P: Pattern + Copy,
    {
        WildString(
            self.0
                .iter()
                .map(|part| match part {
                    WildStringPart::String(s) => WildStringPart::String(s.replace(from, to)),
                    part => part.clone(),
                })
                .collect_vec(),
        )
    }

    pub fn build_acle(&mut self, ctx: &LocalContext) -> Result<(), String> {
        self.build(ctx, TypeRepr::ACLENotation)
    }

    pub fn build_neon_intrinsic_signature(&mut self, ctx: &LocalContext) -> Result<(), String> {
        let repr = TypeRepr::ACLENotation;
        self.iter_mut().try_for_each(|wp| -> Result<(), String> {
            if let WildStringPart::Wildcard(w) = wp {
                match w {
                    &mut Wildcard::NEONType(_, _, ref maybe_suffix_kind) => {
                        if let Some(suffix_kind) = maybe_suffix_kind {
                            let x = ctx.provide_type_wildcard(w).unwrap();
                            *wp = WildStringPart::String(make_neon_suffix(x, *suffix_kind))
                        } else {
                            *wp = WildString::make_default_build(ctx, repr, w)
                        }
                    }
                    _ => *wp = WildString::make_default_build(ctx, repr, w),
                }
            }
            Ok(())
        })
    }

    pub fn build(&mut self, ctx: &LocalContext, repr: TypeRepr) -> Result<(), String> {
        match repr {
            TypeRepr::ACLENotation | TypeRepr::LLVMMachine => {
                self.iter_mut().try_for_each(|wp| -> Result<(), String> {
                    if let WildStringPart::Wildcard(w) = wp {
                        match w {
                            &mut Wildcard::NEONType(_, _, ref maybe_suffix_kind) => {
                                if let Some(suffix_kind) = maybe_suffix_kind {
                                    let x = ctx.provide_type_wildcard(w).unwrap();
                                    *wp = WildStringPart::String(make_neon_suffix(x, *suffix_kind))
                                } else {
                                    *wp = WildString::make_default_build(ctx, repr, w)
                                }
                            }
                            _ => *wp = WildString::make_default_build(ctx, repr, w),
                        }
                    }
                    Ok(())
                })
            }
            _ => self.iter_mut().try_for_each(|wp| -> Result<(), String> {
                if let WildStringPart::Wildcard(w) = wp {
                    *wp = WildString::make_default_build(ctx, repr, w);
                }
                Ok(())
            }),
        }
    }

    fn make_default_build(ctx: &LocalContext, repr: TypeRepr, w: &mut Wildcard) -> WildStringPart {
        WildStringPart::String(
            ctx.provide_substitution_wildcard(w)
                .or_else(|_| ctx.provide_type_wildcard(w).map(|ty| ty.repr(repr)))
                .unwrap(),
        )
    }
}

impl From<String> for WildString {
    fn from(s: String) -> Self {
        WildString(vec![WildStringPart::String(s)])
    }
}

impl FromStr for WildString {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        enum State {
            Normal { start: usize },
            Wildcard { start: usize, count: usize },
            EscapeTokenOpen { start: usize, at: usize },
            EscapeTokenClose { start: usize, at: usize },
        }

        let mut ws = WildString::default();
        match s
            .char_indices()
            .try_fold(State::Normal { start: 0 }, |state, (idx, ch)| {
                match (state, ch) {
                    (State::Normal { start }, '{') => Ok(State::EscapeTokenOpen { start, at: idx }),
                    (State::Normal { start }, '}') => {
                        Ok(State::EscapeTokenClose { start, at: idx })
                    }
                    (State::EscapeTokenOpen { start, at }, '{')
                    | (State::EscapeTokenClose { start, at }, '}') => {
                        if start < at {
                            ws.push_str(&s[start..at])
                        }

                        Ok(State::Normal { start: idx })
                    }
                    (State::EscapeTokenOpen { at, .. }, '}') => Err(format!(
                        "empty wildcard given in string {s:?} at position {at}"
                    )),
                    (State::EscapeTokenOpen { start, at }, _) => {
                        if start < at {
                            ws.push_str(&s[start..at])
                        }

                        Ok(State::Wildcard {
                            start: idx,
                            count: 0,
                        })
                    }
                    (State::EscapeTokenClose { at, .. }, _) => Err(format!(
                        "closing a non-wildcard/bad escape in string {s:?} at position {at}"
                    )),
                    // Nesting wildcards is only supported for `{foo as {bar}}`, wildcards cannot be
                    // nested at the start of a WildString.
                    (State::Wildcard { start, count }, '{') => Ok(State::Wildcard {
                        start,
                        count: count + 1,
                    }),
                    (State::Wildcard { start, count: 0 }, '}') => {
                        ws.push_wildcard(s[start..idx].parse()?);
                        Ok(State::Normal { start: idx + 1 })
                    }
                    (State::Wildcard { start, count }, '}') => Ok(State::Wildcard {
                        start,
                        count: count - 1,
                    }),
                    (state @ State::Normal { .. }, _) | (state @ State::Wildcard { .. }, _) => {
                        Ok(state)
                    }
                }
            })? {
            State::Normal { start } => {
                if start < s.len() {
                    ws.push_str(&s[start..]);
                }

                Ok(ws)
            }
            State::EscapeTokenOpen { at, .. } | State::Wildcard { start: at, .. } => Err(format!(
                "unclosed wildcard in string {s:?} at position {at}"
            )),
            State::EscapeTokenClose { at, .. } => Err(format!(
                "closing a non-wildcard/bad escape in string {s:?} at position {at}"
            )),
        }
    }
}

impl fmt::Display for WildString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .iter()
                .map(|part| match part {
                    WildStringPart::String(s) => s.to_owned(),
                    WildStringPart::Wildcard(w) => format!("{{{w}}}"),
                })
                .join("")
        )
    }
}

impl ToTokens for WildString {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        assert!(
            !self.has_wildcards(),
            "cannot convert string with wildcards {self:?} to TokenStream"
        );
        let str = self.to_string();
        tokens.append_all(quote! { #str })
    }
}

#[cfg(test)]
mod tests {
    use crate::typekinds::*;
    use crate::wildstring::*;

    #[test]
    fn test_empty_string() {
        let ws: WildString = "".parse().unwrap();
        assert_eq!(ws.0.len(), 0);
    }

    #[test]
    fn test_plain_string() {
        let ws: WildString = "plain string".parse().unwrap();
        assert_eq!(ws.0.len(), 1);
        assert_eq!(
            ws,
            WildString(vec![WildStringPart::String("plain string".to_string())])
        )
    }

    #[test]
    fn test_escaped_curly_brackets() {
        let ws: WildString = "VALUE = {{value}}".parse().unwrap();
        assert_eq!(ws.to_string(), "VALUE = {value}");
        assert!(!ws.has_wildcards());
    }

    #[test]
    fn test_escaped_curly_brackets_wildcard() {
        let ws: WildString = "TYPE = {{{type}}}".parse().unwrap();
        assert_eq!(ws.to_string(), "TYPE = {{type}}");
        assert_eq!(ws.0.len(), 4);
        assert!(ws.has_wildcards());
    }

    #[test]
    fn test_wildcard_right_boundary() {
        let s = "string test {type}";
        let ws: WildString = s.parse().unwrap();
        assert_eq!(&ws.to_string(), s);
        assert!(ws.has_wildcards());
    }

    #[test]
    fn test_wildcard_left_boundary() {
        let s = "{type} string test";
        let ws: WildString = s.parse().unwrap();
        assert_eq!(&ws.to_string(), s);
        assert!(ws.has_wildcards());
    }

    #[test]
    fn test_recursive_wildcard() {
        let s = "string test {type[0] as {type[1]}}";
        let ws: WildString = s.parse().unwrap();

        assert_eq!(ws.0.len(), 2);
        assert_eq!(
            ws,
            WildString(vec![
                WildStringPart::String("string test ".to_string()),
                WildStringPart::Wildcard(Wildcard::Scale(
                    Box::new(Wildcard::Type(Some(0))),
                    Box::new(TypeKind::Wildcard(Wildcard::Type(Some(1)))),
                ))
            ])
        );
    }

    #[test]
    fn test_scale_wildcard() {
        let s = "string {type[0] as i8} test";
        let ws: WildString = s.parse().unwrap();

        assert_eq!(ws.0.len(), 3);
        assert_eq!(
            ws,
            WildString(vec![
                WildStringPart::String("string ".to_string()),
                WildStringPart::Wildcard(Wildcard::Scale(
                    Box::new(Wildcard::Type(Some(0))),
                    Box::new(TypeKind::Base(BaseType::Sized(BaseTypeKind::Int, 8))),
                )),
                WildStringPart::String(" test".to_string())
            ])
        );
    }

    #[test]
    fn test_solitaire_wildcard() {
        let ws: WildString = "{type}".parse().unwrap();
        assert_eq!(ws.0.len(), 1);
        assert_eq!(
            ws,
            WildString(vec![WildStringPart::Wildcard(Wildcard::Type(None))])
        )
    }

    #[test]
    fn test_empty_wildcard() {
        "string {}"
            .parse::<WildString>()
            .expect_err("expected parse error");
    }

    #[test]
    fn test_invalid_open_wildcard_right() {
        "string {"
            .parse::<WildString>()
            .expect_err("expected parse error");
    }

    #[test]
    fn test_invalid_close_wildcard_right() {
        "string }"
            .parse::<WildString>()
            .expect_err("expected parse error");
    }

    #[test]
    fn test_invalid_open_wildcard_left() {
        "{string"
            .parse::<WildString>()
            .expect_err("expected parse error");
    }

    #[test]
    fn test_invalid_close_wildcard_left() {
        "}string"
            .parse::<WildString>()
            .expect_err("expected parse error");
    }

    #[test]
    fn test_consecutive_wildcards() {
        let s = "svprf{size_literal[1]}_gather_{type[0]}{index_or_offset}";
        let ws: WildString = s.parse().unwrap();
        assert_eq!(ws.to_string(), s)
    }
}
