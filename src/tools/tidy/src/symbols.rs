//! Ensure that the symbols are sorted and not duplicated within rustc_span/src/symbol.rs

use std::collections::HashSet;
use std::fs::read_to_string;
use std::path::Path;

#[derive(Debug)]
enum Sym<'a> {
    Ident(&'a str),
    IdentLit { ident: &'a str, lit: &'a str },
}

impl Sym<'_> {
    fn symbol(&self, errors: &mut Vec<String>) -> Option<&'_ str> {
        Some(match self {
            Sym::Ident(name) => name,
            Sym::IdentLit { lit, .. } => {
                if lit.contains(|c| matches!(c, '\n' | '\r')) {
                    // FIXME perhaps we could allow escapes by copying logic from `unescape_literal`
                    errors.push(format!("literal contains escapes: {}", lit));
                    return None;
                }
                lit
            }
        })
    }
    fn name(&self) -> &str {
        match self {
            Sym::Ident(name) => name,
            Sym::IdentLit { ident, .. } => ident,
        }
    }
}

pub fn check(compiler_path: &Path, bad: &mut bool) {
    let file = t!(read_to_string(compiler_path.join("rustc_span/src/symbol.rs")));
    let kws = syms(&file, "// keywords-start", "// keywords-end");
    let syms = syms(&file, "// symbols-start", "// symbols-end");
    let mut names: HashSet<_> = "0123456789".chars().map(|c| c.to_string()).collect();
    let mut errors = Vec::new();
    let mut prev: Option<String> = None;
    let mut report = |sym: &Sym<'_>, sorted: bool| {
        let s = if let Some(sym) = sym.symbol(&mut errors) { sym } else { return };
        if !names.insert(s.to_owned()) {
            errors.push(format!("duplicate symbol: `{}`", s));
        }
        if sorted {
            if let Some(prevsym) = prev.take() {
                if &*prevsym > sym.name() {
                    errors.push(format!(
                        "symbol list not sorted: `{}` should come before `{}`",
                        s, prevsym
                    ));
                }
            }
            prev = Some(sym.name().to_string());
        }
    };
    for kw in kws {
        report(&kw, false);
    }
    for sym in syms {
        report(&sym, true);
    }
    if !errors.is_empty() {
        *bad = true;
    }
    for error in errors {
        eprintln!("{error}");
    }
}

fn syms<'a>(
    file: &'a str,
    start_anchor: &'static str,
    end_anchor: &'static str,
) -> impl Iterator<Item = Sym<'a>> + 'a {
    let start = file.find(start_anchor).expect("start anchor") + start_anchor.len();
    let end = file.find(end_anchor).expect("end anchor");
    let symbols = &file[start..end];

    symbols
        .split(",")
        .flat_map(|s| s.lines().map(str::trim).filter(|s| !s.starts_with("//") && !s.is_empty()))
        .map(|s| {
            s.split_once(":").map(|(a, b)| (a.trim(), b.trim())).map_or(
                Sym::Ident(s),
                |(ident, lit)| Sym::IdentLit { ident, lit: &lit[1..lit.len() - 1] },
            )
        })
}
