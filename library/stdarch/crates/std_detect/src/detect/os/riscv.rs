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
        */

        imply!(zbc => zbkc); // defined as subset
        group!(zkn == zbkb & zbkc & zbkx & zkne & zknd & zknh);
        group!(zks == zbkb & zbkc & zbkx & zksed & zksh);
        group!(zk == zkn & zkr & zkt);

        group!(a == zalrsc & zaamo);

        group!(b == zba & zbb & zbs);

        imply!(zhinx => zhinxmin);
        imply!(zdinx | zhinxmin => zfinx);

        imply!(zfh => zfhmin);
        imply!(q => d);
        imply!(d | zfhmin => f);

        imply!(zicntr | zihpm | f | zfinx => zicsr);
        imply!(s | h => zicsr);

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
        // Needs 2 iterations to converge
        // (and 3rd iteration for convergence checking):
        // 1.  [Zk] -> Zkn & Zkr & Zkt
        // 2.  Zkn -> {Zbkb} & {Zbkc} & {Zbkx} & {Zkne} & {Zknd} & {Zknh}
        value.set(Feature::zk as u32);
        let value = imply_features(value);
        assert!(value.test(Feature::zbkb as u32));
        assert!(value.test(Feature::zbkc as u32));
        assert!(value.test(Feature::zbkx as u32));
        assert!(value.test(Feature::zkne as u32));
        assert!(value.test(Feature::zknd as u32));
        assert!(value.test(Feature::zknh as u32));
    }
}
