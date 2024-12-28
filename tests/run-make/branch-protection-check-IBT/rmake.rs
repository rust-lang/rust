// ignore-tidy-linelength
//! A basic smoke test to check for GNU Property Note to see that for `x86_64` targets when [`-Z
//! cf-protection=branch`][intel-cet-tracking-issue] is requested, that the
//!
//! ```text
//! NT_GNU_PROPERTY_TYPE_0 Properties: x86 feature: IBT
//! ```
//!
//! Intel Indirect Branch Tracking (IBT) property is emitted. This was generated in
//! <https://github.com/rust-lang/rust/pull/110304> in order to address
//! <https://github.com/rust-lang/rust/issues/103001>.
//!
//! Note that the precompiled std currently is not compiled with `-Z cf-protection=branch`!
//!
//! In particular, it is expected that:
//!
//! > IBT to only be enabled for the process if `.note.gnu.property` indicates that the executable
//! > was compiled with IBT support and the linker to only tell that IBT is supported if all input
//! > object files indicate that they support IBT, which in turn requires the standard library to be
//! > compiled with IBT enabled.
//!
//! Note that Intel IBT (Indirect Branch Tracking) is not to be confused with Arm's BTI (Branch
//! Target Identification). See below for link to Intel IBT docs.
//!
//! ## Related links
//!
//! - [Tracking Issue for Intel Control Enforcement Technology (CET)][intel-cet-tracking-issue]
//! - Zulip question about this test:
//! <https://rust-lang.zulipchat.com/#narrow/channel/182449-t-compiler.2Fhelp/topic/.E2.9C.94.20Branch.20protection.20and.20.60.2Enote.2Egnu.2Eproperty.60>
//! - Intel IBT docs:
//!   <https://edc.intel.com/content/www/us/en/design/ipla/software-development-platforms/client/platforms/alder-lake-desktop/12th-generation-intel-core-processors-datasheet-volume-1-of-2/006/indirect-branch-tracking/>
//!
//! [intel-cet-tracking-issue]: https://github.com/rust-lang/rust/issues/93754

//@ needs-llvm-components: x86

// FIXME(#93754): increase the test coverage of this test.
//@ only-x86_64-unknown-linux-gnu
//@ ignore-cross-compile

use run_make_support::{bare_rustc, llvm_readobj};

fn main() {
    // `main.rs` is `#![no_std]` to not pull in the currently not-compiled-with-IBT precompiled std.
    bare_rustc()
        .input("main.rs")
        .target("x86_64-unknown-linux-gnu")
        .arg("-Zcf-protection=branch")
        .arg("-Clink-args=-nostartfiles")
        .run();

    llvm_readobj().arg("-nW").input("main").run().assert_stdout_contains(".note.gnu.property");
}
