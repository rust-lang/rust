// Checks that all the targets returned by `rustc --print target-list` are valid target
// specifications.

// ignore-tidy-linelength
//@ needs-llvm-components: aarch64 arm avr bpf csky hexagon loongarch m68k mips msp430 nvptx powerpc riscv sparc systemz webassembly x86
// FIXME(jieyouxu): there has to be a better way to do this, without the needs-llvm-components it
// will fail on LLVM built without all of the components listed above.

use run_make_support::bare_rustc;

// FIXME(#127877): certain experimental targets fail with creating a 'LLVM TargetMachine' in CI, so
// we skip them.
const EXPERIMENTAL_TARGETS: &[&str] = &["avr", "m68k", "csky", "xtensa"];

fn main() {
    let targets = bare_rustc().print("target-list").run().stdout_utf8();

    for target in targets.lines() {
        // skip experimental targets that would otherwise fail
        if EXPERIMENTAL_TARGETS.iter().any(|experimental| target.contains(experimental)) {
            continue;
        }

        bare_rustc().target(target).print("sysroot").run();
    }
}
