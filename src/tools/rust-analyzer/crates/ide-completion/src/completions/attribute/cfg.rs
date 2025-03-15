//! Completion for cfg

use ide_db::SymbolKind;
use itertools::Itertools;
use syntax::{AstToken, Direction, NodeOrToken, SyntaxKind, algo, ast::Ident};

use crate::{CompletionItem, completions::Completions, context::CompletionContext};

pub(crate) fn complete_cfg(acc: &mut Completions, ctx: &CompletionContext<'_>) {
    let add_completion = |item: &str| {
        let mut completion =
            CompletionItem::new(SymbolKind::BuiltinAttr, ctx.source_range(), item, ctx.edition);
        completion.insert_text(format!(r#""{item}""#));
        acc.add(completion.build(ctx.db));
    };

    // FIXME: Move this into context/analysis.rs
    let previous = ctx
        .original_token
        .prev_token()
        .and_then(|it| {
            if matches!(it.kind(), SyntaxKind::EQ) {
                Some(it.into())
            } else {
                algo::non_trivia_sibling(it.into(), Direction::Prev)
            }
        })
        .filter(|t| matches!(t.kind(), SyntaxKind::EQ))
        .and_then(|it| algo::non_trivia_sibling(it.prev_sibling_or_token()?, Direction::Prev))
        .map(|it| match it {
            NodeOrToken::Node(_) => None,
            NodeOrToken::Token(t) => Ident::cast(t),
        });
    match previous {
        Some(None) => (),
        Some(Some(p)) => match p.text() {
            "target_arch" => KNOWN_ARCH.iter().copied().for_each(add_completion),
            "target_env" => KNOWN_ENV.iter().copied().for_each(add_completion),
            "target_os" => KNOWN_OS.iter().copied().for_each(add_completion),
            "target_vendor" => KNOWN_VENDOR.iter().copied().for_each(add_completion),
            "target_endian" => ["little", "big"].into_iter().for_each(add_completion),
            name => ctx.krate.potential_cfg(ctx.db).get_cfg_values(name).cloned().for_each(|s| {
                let s = s.as_str();
                let insert_text = format!(r#""{s}""#);
                let mut item = CompletionItem::new(
                    SymbolKind::BuiltinAttr,
                    ctx.source_range(),
                    s,
                    ctx.edition,
                );
                item.insert_text(insert_text);

                acc.add(item.build(ctx.db));
            }),
        },
        None => ctx.krate.potential_cfg(ctx.db).get_cfg_keys().cloned().unique().for_each(|s| {
            let s = s.as_str();
            let item =
                CompletionItem::new(SymbolKind::BuiltinAttr, ctx.source_range(), s, ctx.edition);
            acc.add(item.build(ctx.db));
        }),
    }
}

const KNOWN_ARCH: [&str; 20] = [
    "aarch64",
    "arm",
    "avr",
    "csky",
    "hexagon",
    "mips",
    "mips64",
    "msp430",
    "nvptx64",
    "powerpc",
    "powerpc64",
    "riscv32",
    "riscv64",
    "s390x",
    "sparc",
    "sparc64",
    "wasm32",
    "wasm64",
    "x86",
    "x86_64",
];

const KNOWN_ENV: [&str; 7] = ["eabihf", "gnu", "gnueabihf", "msvc", "relibc", "sgx", "uclibc"];

const KNOWN_OS: [&str; 20] = [
    "cuda",
    "dragonfly",
    "emscripten",
    "freebsd",
    "fuchsia",
    "haiku",
    "hermit",
    "illumos",
    "l4re",
    "linux",
    "netbsd",
    "none",
    "openbsd",
    "psp",
    "redox",
    "solaris",
    "uefi",
    "unknown",
    "vxworks",
    "windows",
];

const KNOWN_VENDOR: [&str; 8] =
    ["apple", "fortanix", "nvidia", "pc", "sony", "unknown", "wrs", "uwp"];
