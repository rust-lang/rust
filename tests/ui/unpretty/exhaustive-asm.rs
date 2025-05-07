//@ revisions: expanded hir
//@[expanded]compile-flags: -Zunpretty=expanded
//@[expanded]check-pass
//@[hir]compile-flags: -Zunpretty=hir
//@[hir]check-pass
//@ edition:2024
//@ only-x86_64
//
// asm parts of exhaustive.rs. Separate because we only run this on x86_64.

mod expressions {
    /// ExprKind::InlineAsm
    fn expr_inline_asm() {
        let x;
        core::arch::asm!(
            "mov {tmp}, {x}",
            "shl {tmp}, 1",
            "shl {x}, 2",
            "add {x}, {tmp}",
            x = inout(reg) x,
            tmp = out(reg) _,
        );
    }
}

mod items {
    /// ItemKind::GlobalAsm
    mod item_global_asm {
        core::arch::global_asm!(".globl my_asm_func");
    }
}
