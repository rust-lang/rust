//! The settings of `-Z raise-optimize-speed`, the experimental option that makes
//! `#[optimize(speed)]` take effect in a crate built at `-C opt-level=0`.

use crate::config::OptLevel;

/// Which modules of a crate built at `-C opt-level=0` get the optimizing pipeline for the sake
/// of its `#[optimize(speed)]` functions; the first setting of `-Z raise-optimize-speed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RaiseOptimizeSpeedForm {
    /// `off`: none. `#[optimize(speed)]` has no effect in such a crate.
    #[default]
    Off,
    /// `crate`: every module, with `optnone` on every function that lacks the attribute.
    Crate,
    /// `unit`: as `crate`, but only the modules that hold a function with the attribute.
    Unit,
    /// `one-module`: one module made for the purpose, which holds every function with the
    /// attribute and nothing that lacks it.
    OneModule,
    /// `modules`: the modules that the functions with the attribute are placed in when no such
    /// function may share a module with a function that lacks it.
    Modules,
}

impl RaiseOptimizeSpeedForm {
    /// Whether the functions with the attribute stay in the modules they would be in anyway, so
    /// that the functions beside them must be kept unoptimized by `optnone`.
    pub fn is_in_place(self) -> bool {
        matches!(self, RaiseOptimizeSpeedForm::Crate | RaiseOptimizeSpeedForm::Unit)
    }

    /// Whether the functions with the attribute get modules to themselves.
    pub fn separates_modules(self) -> bool {
        matches!(self, RaiseOptimizeSpeedForm::OneModule | RaiseOptimizeSpeedForm::Modules)
    }
}

/// Whether the compiler gives a crate a function with `#[optimize(speed)]` of its own accord, so
/// that the cost of holding one can be measured on crates that have none; the `probe=` setting
/// of `-Z raise-optimize-speed`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RaiseOptimizeSpeedProbe {
    /// No function is supplied.
    #[default]
    Off,
    /// `probe=added-unused`: an empty function with the attribute is added to the crate, and
    /// nothing refers to it.
    AddedUnused,
    /// `probe=added`: the same function is added with a `#[used]` static that refers to it, so
    /// that a program which links the crate loads the function's module.
    Added,
    /// `probe=chosen`: the first function the crate defines and emits code for is treated as
    /// having the attribute.
    Chosen,
}

/// Settings for the `-Z raise-optimize-speed` flag, which makes `#[optimize(speed)]` take
/// effect in a crate built at `-C opt-level=0`.
#[derive(Clone, Copy, Debug, PartialEq, Hash)]
pub struct RaiseOptimizeSpeed {
    /// `off`, `crate`, `unit`, `one-module`, or `modules`.
    pub form: RaiseOptimizeSpeedForm,
    /// `level=2` or `level=3`: the level that a raised module is optimized at.
    pub level: OptLevel,
    /// `noipa`: with an in-place form, also mark the functions that lack the attribute `noipa`.
    pub noipa: bool,
    /// `copy-callees`: with a separating form, give a raised module private copies of the generic
    /// and `#[inline]` functions that its functions call.
    pub copy_callees: bool,
    /// `mir`: run the MIR optimizations of an optimized build, the MIR inliner among them, on the
    /// functions with the attribute.
    pub mir: bool,
    /// `import`: with a separating form, let the raised modules import functions from the
    /// crate's other modules.
    pub import: bool,
    /// `codegen-level`: with a separating form, emit the IR of a raised module as an optimized
    /// build would.
    pub codegen_level: bool,
    /// `probe=added-unused`, `probe=added`, or `probe=chosen`.
    pub probe: RaiseOptimizeSpeedProbe,
}

impl Default for RaiseOptimizeSpeed {
    fn default() -> Self {
        RaiseOptimizeSpeed {
            form: RaiseOptimizeSpeedForm::Off,
            level: OptLevel::Aggressive,
            noipa: false,
            copy_callees: false,
            mir: false,
            import: false,
            codegen_level: false,
            probe: RaiseOptimizeSpeedProbe::Off,
        }
    }
}
