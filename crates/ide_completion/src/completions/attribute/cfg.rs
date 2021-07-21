//! Completion for cfg

use std::iter;

use syntax::SyntaxKind;

use crate::{
    completions::Completions, context::CompletionContext, item::CompletionKind, CompletionItem,
    CompletionItemKind,
};

pub(crate) fn complete_cfg(acc: &mut Completions, ctx: &CompletionContext) {
    let add_completion = |item: &&str| {
        let mut completion =
            CompletionItem::new(CompletionKind::Attribute, ctx.source_range(), *item);
        completion.insert_text(format!(r#""{}""#, item));
        completion.kind(CompletionItemKind::Attribute);
        acc.add(completion.build());
    };

    let previous = iter::successors(ctx.original_token.prev_token(), |t| {
        (matches!(t.kind(), SyntaxKind::EQ) || t.kind().is_trivia())
            .then(|| t.prev_token())
            .flatten()
    })
    .find(|t| matches!(t.kind(), SyntaxKind::IDENT));

    match previous.as_ref().map(|p| p.text()) {
        Some("target_arch") => KNOWN_ARCH.iter().for_each(add_completion),
        Some("target_env") => KNOWN_ENV.iter().for_each(add_completion),
        Some("target_os") => KNOWN_OS.iter().for_each(add_completion),
        Some("target_vendor") => KNOWN_VENDOR.iter().for_each(add_completion),
        Some("target_endian") => ["little", "big"].iter().for_each(add_completion),
        Some(name) => {
            if let Some(krate) = ctx.krate {
                krate.potential_cfg(ctx.db).get_cfg_values(&name).iter().for_each(|s| {
                    let mut item = CompletionItem::new(
                        CompletionKind::Attribute,
                        ctx.source_range(),
                        s.as_str(),
                    );
                    item.insert_text(format!(r#""{}""#, s));

                    acc.add(item.build());
                })
            };
        }
        None => {
            if let Some(krate) = ctx.krate {
                krate.potential_cfg(ctx.db).get_cfg_keys().iter().for_each(|s| {
                    let item = CompletionItem::new(
                        CompletionKind::Attribute,
                        ctx.source_range(),
                        s.as_str(),
                    );
                    acc.add(item.build());
                })
            }
        }
    };
}

const KNOWN_ARCH: [&str; 19] = [
    "aarch64",
    "arm",
    "avr",
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
