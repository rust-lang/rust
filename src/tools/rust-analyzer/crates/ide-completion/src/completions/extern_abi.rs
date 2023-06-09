//! Completes function abi strings.
use syntax::{
    ast::{self, IsString},
    AstNode, AstToken,
};

use crate::{
    completions::Completions, context::CompletionContext, CompletionItem, CompletionItemKind,
};

// Most of these are feature gated, we should filter/add feature gate completions once we have them.
const SUPPORTED_CALLING_CONVENTIONS: &[&str] = &[
    "Rust",
    "C",
    "C-unwind",
    "cdecl",
    "stdcall",
    "stdcall-unwind",
    "fastcall",
    "vectorcall",
    "thiscall",
    "thiscall-unwind",
    "aapcs",
    "win64",
    "sysv64",
    "ptx-kernel",
    "msp430-interrupt",
    "x86-interrupt",
    "amdgpu-kernel",
    "efiapi",
    "avr-interrupt",
    "avr-non-blocking-interrupt",
    "C-cmse-nonsecure-call",
    "wasm",
    "system",
    "system-unwind",
    "rust-intrinsic",
    "rust-call",
    "platform-intrinsic",
    "unadjusted",
];

pub(crate) fn complete_extern_abi(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    expanded: &ast::String,
) -> Option<()> {
    if !expanded.syntax().parent().map_or(false, |it| ast::Abi::can_cast(it.kind())) {
        return None;
    }
    let abi_str = expanded;
    let source_range = abi_str.text_range_between_quotes()?;
    for &abi in SUPPORTED_CALLING_CONVENTIONS {
        CompletionItem::new(CompletionItemKind::Keyword, source_range, abi).add_to(acc, ctx.db);
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::tests::{check_edit, completion_list_no_kw};

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = completion_list_no_kw(ra_fixture);
        expect.assert_eq(&actual);
    }

    #[test]
    fn only_completes_in_string_literals() {
        check(
            r#"
$0 fn foo {}
"#,
            expect![[]],
        );
    }

    #[test]
    fn requires_extern_prefix() {
        check(
            r#"
"$0" fn foo {}
"#,
            expect![[]],
        );
    }

    #[test]
    fn works() {
        check(
            r#"
extern "$0" fn foo {}
"#,
            expect![[]],
        );
        check_edit(
            "Rust",
            r#"
extern "$0" fn foo {}
"#,
            r#"
extern "Rust" fn foo {}
"#,
        );
    }
}
