//@ check-pass
//@ needs-asm-support
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

fn invoke(pc_section: &[usize]) {
    unsafe {
        std::arch::asm!(
            "/* {} */",
            in(reg) pc_section[0]
        );
    }
}

fn main() {}
