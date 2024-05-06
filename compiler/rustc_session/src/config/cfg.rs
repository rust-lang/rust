//! cfg and check-cfg configuration
//!
//! This module contains the definition of [`Cfg`] and [`CheckCfg`]
//! as well as the logic for creating the default configuration for a
//! given [`Session`].
//!
//! It also contains the filling of the well known configs, which should
//! ALWAYS be in sync with the default_configuration.
//!
//! ## Adding a new cfg
//!
//! Adding a new feature requires two new symbols one for the cfg it-self
//! and the second one for the unstable feature gate, those are defined in
//! `rustc_span::symbol`.
//!
//! As well as the following points,
//!  - Add the activation logic in [`default_configuration`]
//!  - Add the cfg to [`CheckCfg::fill_well_known`] (and related files),
//!    so that the compiler can know the cfg is expected
//!  - Add the feature gating in `compiler/rustc_feature/src/builtin_attrs.rs`

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexSet};
use rustc_span::symbol::{sym, Symbol};
use rustc_target::abi::Align;
use rustc_target::spec::{PanicStrategy, RelocModel, SanitizerSet};
use rustc_target::spec::{Target, TargetTriple, TARGETS};

use crate::config::CrateType;
use crate::Session;

use std::hash::Hash;
use std::iter;

/// The parsed `--cfg` options that define the compilation environment of the
/// crate, used to drive conditional compilation.
///
/// An `FxIndexSet` is used to ensure deterministic ordering of error messages
/// relating to `--cfg`.
pub type Cfg = FxIndexSet<(Symbol, Option<Symbol>)>;

/// The parsed `--check-cfg` options.
#[derive(Default)]
pub struct CheckCfg {
    /// Is well known names activated
    pub exhaustive_names: bool,
    /// Is well known values activated
    pub exhaustive_values: bool,
    /// All the expected values for a config name
    pub expecteds: FxHashMap<Symbol, ExpectedValues<Symbol>>,
    /// Well known names (only used for diagnostics purposes)
    pub well_known_names: FxHashSet<Symbol>,
}

pub enum ExpectedValues<T> {
    Some(FxHashSet<Option<T>>),
    Any,
}

impl<T: Eq + Hash> ExpectedValues<T> {
    fn insert(&mut self, value: T) -> bool {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.insert(Some(value)),
            ExpectedValues::Any => false,
        }
    }
}

impl<T: Eq + Hash> Extend<T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(Some)),
            ExpectedValues::Any => {}
        }
    }
}

impl<'a, T: Eq + Hash + Copy + 'a> Extend<&'a T> for ExpectedValues<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        match self {
            ExpectedValues::Some(expecteds) => expecteds.extend(iter.into_iter().map(|a| Some(*a))),
            ExpectedValues::Any => {}
        }
    }
}

/// Generate the default configs for a given session
pub(crate) fn default_configuration(sess: &Session) -> Cfg {
    let mut ret = Cfg::default();

    macro_rules! ins_none {
        ($key:expr) => {
            ret.insert(($key, None));
        };
    }
    macro_rules! ins_str {
        ($key:expr, $val_str:expr) => {
            ret.insert(($key, Some(Symbol::intern($val_str))));
        };
    }
    macro_rules! ins_sym {
        ($key:expr, $val_sym:expr) => {
            ret.insert(($key, Some($val_sym)));
        };
    }

    // Symbols are inserted in alphabetical order as much as possible.
    // The exceptions are where control flow forces things out of order.
    //
    // Run `rustc --print cfg` to see the configuration in practice.
    //
    // NOTE: These insertions should be kept in sync with
    // `CheckCfg::fill_well_known` below.

    if sess.opts.debug_assertions {
        ins_none!(sym::debug_assertions);
    }

    if sess.overflow_checks() {
        ins_none!(sym::overflow_checks);
    }

    ins_sym!(sym::panic, sess.panic_strategy().desc_symbol());

    // JUSTIFICATION: before wrapper fn is available
    #[allow(rustc::bad_opt_access)]
    if sess.opts.crate_types.contains(&CrateType::ProcMacro) {
        ins_none!(sym::proc_macro);
    }

    if sess.is_nightly_build() {
        ins_sym!(sym::relocation_model, sess.target.relocation_model.desc_symbol());
    }

    for mut s in sess.opts.unstable_opts.sanitizer {
        // KASAN is still ASAN under the hood, so it uses the same attribute.
        if s == SanitizerSet::KERNELADDRESS {
            s = SanitizerSet::ADDRESS;
        }
        ins_str!(sym::sanitize, &s.to_string());
    }

    if sess.is_sanitizer_cfi_generalize_pointers_enabled() {
        ins_none!(sym::sanitizer_cfi_generalize_pointers);
    }
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        ins_none!(sym::sanitizer_cfi_normalize_integers);
    }

    ins_str!(sym::target_abi, &sess.target.abi);
    ins_str!(sym::target_arch, &sess.target.arch);
    ins_str!(sym::target_endian, sess.target.endian.as_str());
    ins_str!(sym::target_env, &sess.target.env);

    for family in sess.target.families.as_ref() {
        ins_str!(sym::target_family, family);
        if family == "windows" {
            ins_none!(sym::windows);
        } else if family == "unix" {
            ins_none!(sym::unix);
        }
    }

    // `target_has_atomic*`
    let layout = sess.target.parse_data_layout().unwrap_or_else(|err| {
        sess.dcx().emit_fatal(err);
    });
    let mut has_atomic = false;
    for (i, align) in [
        (8, layout.i8_align.abi),
        (16, layout.i16_align.abi),
        (32, layout.i32_align.abi),
        (64, layout.i64_align.abi),
        (128, layout.i128_align.abi),
    ] {
        if i >= sess.target.min_atomic_width() && i <= sess.target.max_atomic_width() {
            if !has_atomic {
                has_atomic = true;
                if sess.is_nightly_build() {
                    if sess.target.atomic_cas {
                        ins_none!(sym::target_has_atomic);
                    }
                    ins_none!(sym::target_has_atomic_load_store);
                }
            }
            let mut insert_atomic = |sym, align: Align| {
                if sess.target.atomic_cas {
                    ins_sym!(sym::target_has_atomic, sym);
                }
                if align.bits() == i {
                    ins_sym!(sym::target_has_atomic_equal_alignment, sym);
                }
                ins_sym!(sym::target_has_atomic_load_store, sym);
            };
            insert_atomic(sym::integer(i), align);
            if sess.target.pointer_width as u64 == i {
                insert_atomic(sym::ptr, layout.pointer_align.abi);
            }
        }
    }

    ins_str!(sym::target_os, &sess.target.os);
    ins_sym!(sym::target_pointer_width, sym::integer(sess.target.pointer_width));

    if sess.opts.unstable_opts.has_thread_local.unwrap_or(sess.target.has_thread_local) {
        ins_none!(sym::target_thread_local);
    }

    ins_str!(sym::target_vendor, &sess.target.vendor);

    // If the user wants a test runner, then add the test cfg.
    if sess.is_test_crate() {
        ins_none!(sym::test);
    }

    if sess.ub_checks() {
        ins_none!(sym::ub_checks);
    }

    ret
}

impl CheckCfg {
    /// Fill the current [`CheckCfg`] with all the well known cfgs
    pub fn fill_well_known(&mut self, current_target: &Target) {
        if !self.exhaustive_values && !self.exhaustive_names {
            return;
        }

        // for `#[cfg(foo)]` (ie. cfg value is none)
        let no_values = || {
            let mut values = FxHashSet::default();
            values.insert(None);
            ExpectedValues::Some(values)
        };

        // preparation for inserting some values
        let empty_values = || {
            let values = FxHashSet::default();
            ExpectedValues::Some(values)
        };

        macro_rules! ins {
            ($name:expr, $values:expr) => {{
                self.well_known_names.insert($name);
                self.expecteds.entry($name).or_insert_with($values)
            }};
        }

        // Symbols are inserted in alphabetical order as much as possible.
        // The exceptions are where control flow forces things out of order.
        //
        // NOTE: This should be kept in sync with `default_configuration`.
        // Note that symbols inserted conditionally in `default_configuration`
        // are inserted unconditionally here.
        //
        // When adding a new config here you should also update
        // `tests/ui/check-cfg/well-known-values.rs` (in order to test the
        // expected values of the new config) and bless the all directory.
        //
        // Don't forget to update `src/doc/rustc/src/check-cfg.md`
        // in the unstable book as well!

        ins!(sym::debug_assertions, no_values);

        // These four are never set by rustc, but we set them anyway; they
        // should not trigger the lint because `cargo clippy`, `cargo doc`,
        // `cargo test`, `cargo miri run` and `cargo fmt` (respectively)
        // can set them.
        ins!(sym::clippy, no_values);
        ins!(sym::doc, no_values);
        ins!(sym::doctest, no_values);
        ins!(sym::miri, no_values);
        ins!(sym::rustfmt, no_values);

        ins!(sym::overflow_checks, no_values);

        ins!(sym::panic, empty_values).extend(&PanicStrategy::all());

        ins!(sym::proc_macro, no_values);

        ins!(sym::relocation_model, empty_values).extend(RelocModel::all());

        let sanitize_values = SanitizerSet::all()
            .into_iter()
            .map(|sanitizer| Symbol::intern(sanitizer.as_str().unwrap()));
        ins!(sym::sanitize, empty_values).extend(sanitize_values);

        ins!(sym::sanitizer_cfi_generalize_pointers, no_values);
        ins!(sym::sanitizer_cfi_normalize_integers, no_values);

        ins!(sym::target_feature, empty_values).extend(
            rustc_target::target_features::all_known_features()
                .map(|(f, _sb)| f)
                .chain(rustc_target::target_features::RUSTC_SPECIFIC_FEATURES.iter().cloned())
                .map(Symbol::intern),
        );

        // sym::target_*
        {
            const VALUES: [&Symbol; 8] = [
                &sym::target_abi,
                &sym::target_arch,
                &sym::target_endian,
                &sym::target_env,
                &sym::target_family,
                &sym::target_os,
                &sym::target_pointer_width,
                &sym::target_vendor,
            ];

            // Initialize (if not already initialized)
            for &e in VALUES {
                if !self.exhaustive_values {
                    ins!(e, || ExpectedValues::Any);
                } else {
                    ins!(e, empty_values);
                }
            }

            if self.exhaustive_values {
                // Get all values map at once otherwise it would be costly.
                // (8 values * 220 targets ~= 1760 times, at the time of writing this comment).
                let [
                    values_target_abi,
                    values_target_arch,
                    values_target_endian,
                    values_target_env,
                    values_target_family,
                    values_target_os,
                    values_target_pointer_width,
                    values_target_vendor,
                ] = self
                    .expecteds
                    .get_many_mut(VALUES)
                    .expect("unable to get all the check-cfg values buckets");

                for target in TARGETS
                    .iter()
                    .map(|target| Target::expect_builtin(&TargetTriple::from_triple(target)))
                    .chain(iter::once(current_target.clone()))
                {
                    values_target_abi.insert(Symbol::intern(&target.options.abi));
                    values_target_arch.insert(Symbol::intern(&target.arch));
                    values_target_endian.insert(Symbol::intern(target.options.endian.as_str()));
                    values_target_env.insert(Symbol::intern(&target.options.env));
                    values_target_family.extend(
                        target.options.families.iter().map(|family| Symbol::intern(family)),
                    );
                    values_target_os.insert(Symbol::intern(&target.options.os));
                    values_target_pointer_width.insert(sym::integer(target.pointer_width));
                    values_target_vendor.insert(Symbol::intern(&target.options.vendor));
                }
            }
        }

        let atomic_values = &[
            sym::ptr,
            sym::integer(8usize),
            sym::integer(16usize),
            sym::integer(32usize),
            sym::integer(64usize),
            sym::integer(128usize),
        ];
        for sym in [
            sym::target_has_atomic,
            sym::target_has_atomic_equal_alignment,
            sym::target_has_atomic_load_store,
        ] {
            ins!(sym, no_values).extend(atomic_values);
        }

        ins!(sym::target_thread_local, no_values);

        ins!(sym::test, no_values);

        ins!(sym::ub_checks, no_values);

        ins!(sym::unix, no_values);
        ins!(sym::windows, no_values);
    }
}
