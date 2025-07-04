//! Run-time feature detection utility for RISC-V.
//!
//! On RISC-V, full feature detection needs a help of one or more
//! feature detection mechanisms (usually provided by the operating system).
//!
//! RISC-V architecture defines many extensions and some have dependency to others.
//! More importantly, some of them cannot be enabled without resolving such
//! dependencies due to limited set of features that such mechanisms provide.
//!
//! This module provides an OS-independent utility to process such relations
//! between RISC-V extensions.

use crate::detect::{Feature, cache};

/// Imply features by the given set of enabled features.
///
/// Note that it does not perform any consistency checks including existence of
/// conflicting extensions and/or complicated requirements.  Eliminating such
/// inconsistencies is the responsibility of the feature detection logic and
/// its provider(s).
pub(crate) fn imply_features(mut value: cache::Initializer) -> cache::Initializer {
    loop {
        // Check convergence of the feature flags later.
        let prev = value;

        // Expect that the optimizer turns repeated operations into
        // a fewer number of bit-manipulation operations.
        macro_rules! imply {
            // Regular implication:
            // A1 => (B1[, B2...]), A2 => (B1[, B2...]) and so on.
            ($($from: ident)|+ => $($to: ident)&+) => {
                if [$(Feature::$from as u32),+].iter().any(|&x| value.test(x)) {
                    $(
                        value.set(Feature::$to as u32);
                    )+
                }
            };
            // Implication with multiple requirements:
            // A1 && A2 ... => (B1[, B2...]).
            ($($from: ident)&+ => $($to: ident)&+) => {
                if [$(Feature::$from as u32),+].iter().all(|&x| value.test(x)) {
                    $(
                        value.set(Feature::$to as u32);
                    )+
                }
            };
        }
        macro_rules! group {
            ($group: ident == $($member: ident)&+) => {
                // Forward implication as defined in the specifications.
                imply!($group => $($member)&+);
                // Reverse implication to "group extension" from its members.
                // This is not a part of specifications but convenient for
                // feature detection and implemented in e.g. LLVM.
                imply!($($member)&+ => $group);
            };
        }

        /*
            If a dependency/implication is not explicitly stated in the
            specification, it is denoted as a comment as follows:
            "defined as subset":
                The latter extension is described as a subset of the former
                (but the evidence is weak).
            "functional":
                The former extension is functionally a superset of the latter
                (no direct references though).
        */

        imply!(zvbb => zvkb);

        // Certain set of vector cryptography extensions form a group.
        group!(zvkn == zvkned & zvknhb & zvkb & zvkt);
        group!(zvknc == zvkn & zvbc);
        group!(zvkng == zvkn & zvkg);
        group!(zvks == zvksed & zvksh & zvkb & zvkt);
        group!(zvksc == zvks & zvbc);
        group!(zvksg == zvks & zvkg);

        imply!(zvknhb => zvknha); // functional

        // For vector cryptography, Zvknhb and Zvbc require integer arithmetic
        // with EEW=64 (Zve64x) while others not depending on them
        // require EEW=32 (Zve32x).
        imply!(zvknhb | zvbc => zve64x);
        imply!(zvbb | zvkb | zvkg | zvkned | zvknha | zvksed | zvksh => zve32x);

        imply!(zbc => zbkc); // defined as subset
        group!(zkn == zbkb & zbkc & zbkx & zkne & zknd & zknh);
        group!(zks == zbkb & zbkc & zbkx & zksed & zksh);
        group!(zk == zkn & zkr & zkt);

        imply!(zacas => zaamo);
        group!(a == zalrsc & zaamo);

        group!(b == zba & zbb & zbs);

        imply!(zcf => zca & f);
        imply!(zcd => zca & d);
        imply!(zcmop | zcb => zca);

        imply!(zhinx => zhinxmin);
        imply!(zdinx | zhinxmin => zfinx);

        imply!(zvfh => zvfhmin); // functional
        imply!(zvfh => zve32f & zfhmin);
        imply!(zvfhmin => zve32f);
        imply!(zvfbfwma => zvfbfmin & zfbfmin);
        imply!(zvfbfmin => zve32f);

        imply!(v => zve64d);
        imply!(zve64d => zve64f & d);
        imply!(zve64f => zve64x & zve32f);
        imply!(zve64x => zve32x);
        imply!(zve32f => zve32x & f);

        imply!(zfh => zfhmin);
        imply!(q => d);
        imply!(d | zfhmin | zfa => f);
        imply!(zfbfmin => f); // and some of (not all) "Zfh" instructions.

        // Relatively complex implication rules from the "C" extension.
        imply!(c => zca);
        imply!(c & d => zcd);
        #[cfg(target_arch = "riscv32")]
        imply!(c & f => zcf);

        imply!(zicntr | zihpm | f | zfinx | zve32x => zicsr);

        // Loop until the feature flags converge.
        if prev == value {
            return value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_direct() {
        let mut value = cache::Initializer::default();
        value.set(Feature::f as u32);
        // F (and other extensions with CSRs) -> Zicsr
        assert!(imply_features(value).test(Feature::zicsr as u32));
    }

    #[test]
    fn simple_indirect() {
        let mut value = cache::Initializer::default();
        value.set(Feature::q as u32);
        // Q -> D, D -> F, F -> Zicsr
        assert!(imply_features(value).test(Feature::zicsr as u32));
    }

    #[test]
    fn complex_zcd() {
        let mut value = cache::Initializer::default();
        // C & D -> Zcd
        value.set(Feature::c as u32);
        assert!(!imply_features(value).test(Feature::zcd as u32));
        value.set(Feature::d as u32);
        assert!(imply_features(value).test(Feature::zcd as u32));
    }

    #[test]
    fn group_simple_forward() {
        let mut value = cache::Initializer::default();
        // A -> Zalrsc & Zaamo (forward implication)
        value.set(Feature::a as u32);
        let value = imply_features(value);
        assert!(value.test(Feature::zalrsc as u32));
        assert!(value.test(Feature::zaamo as u32));
    }

    #[test]
    fn group_simple_backward() {
        let mut value = cache::Initializer::default();
        // Zalrsc & Zaamo -> A (reverse implication)
        value.set(Feature::zalrsc as u32);
        value.set(Feature::zaamo as u32);
        assert!(imply_features(value).test(Feature::a as u32));
    }

    #[test]
    fn group_complex_convergence() {
        let mut value = cache::Initializer::default();
        // Needs 3 iterations to converge
        // (and 4th iteration for convergence checking):
        // 1.  [Zvksc] -> Zvks & Zvbc
        // 2.  Zvks -> Zvksed & Zvksh & Zvkb & Zvkt
        // 3a. [Zvkned] & [Zvknhb] & [Zvkb] & Zvkt -> {Zvkn}
        // 3b. Zvkn & Zvbc -> {Zvknc}
        value.set(Feature::zvksc as u32);
        value.set(Feature::zvkned as u32);
        value.set(Feature::zvknhb as u32);
        value.set(Feature::zvkb as u32);
        let value = imply_features(value);
        assert!(value.test(Feature::zvkn as u32));
        assert!(value.test(Feature::zvknc as u32));
    }
}
