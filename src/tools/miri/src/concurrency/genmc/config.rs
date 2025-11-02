use genmc_sys::LogLevel;
use rustc_abi::Endian;
use rustc_middle::ty::TyCtxt;

use super::GenmcParams;
use crate::{IsolatedOp, MiriConfig, RejectOpWith};

/// Configuration for GenMC mode.
/// The `params` field is shared with the C++ side.
/// The remaining options are kept on the Rust side.
#[derive(Debug, Default, Clone)]
pub struct GenmcConfig {
    /// Parameters sent to the C++ side to create a new handle to the GenMC model checker.
    pub(super) params: GenmcParams,
    pub(super) do_estimation: bool,
    /// Print the output message that GenMC generates when an error occurs.
    /// This error message is currently hard to use, since there is no clear mapping between the events that GenMC sees and the Rust code location where this event was produced.
    pub(super) print_genmc_output: bool,
    /// The log level for GenMC.
    pub(super) log_level: LogLevel,
    /// Enable more verbose output, such as number of executions estimate
    /// and time to completion of verification step.
    pub(super) verbose_output: bool,
}

impl GenmcConfig {
    /// Function for parsing command line options for GenMC mode.
    ///
    /// All GenMC arguments start with the string "-Zmiri-genmc".
    /// Passing any GenMC argument will enable GenMC mode.
    ///
    /// `trimmed_arg` should be the argument to be parsed, with the suffix "-Zmiri-genmc" removed.
    pub fn parse_arg(
        genmc_config: &mut Option<GenmcConfig>,
        trimmed_arg: &str,
    ) -> Result<(), String> {
        if genmc_config.is_none() {
            *genmc_config = Some(Default::default());
        }
        if trimmed_arg.is_empty() {
            return Ok(()); // this corresponds to "-Zmiri-genmc"
        }
        let genmc_config = genmc_config.as_mut().unwrap();
        let Some(trimmed_arg) = trimmed_arg.strip_prefix("-") else {
            return Err(format!("Invalid GenMC argument \"-Zmiri-genmc{trimmed_arg}\""));
        };
        if let Some(log_level) = trimmed_arg.strip_prefix("log=") {
            genmc_config.log_level = log_level.parse()?;
        } else if let Some(trimmed_arg) = trimmed_arg.strip_prefix("print-exec-graphs") {
            use genmc_sys::ExecutiongraphPrinting;
            genmc_config.params.print_execution_graphs = match trimmed_arg {
                "=none" => ExecutiongraphPrinting::None,
                // Make GenMC print explored executions.
                "" | "=explored" => ExecutiongraphPrinting::Explored,
                // Make GenMC print blocked executions.
                "=blocked" => ExecutiongraphPrinting::Blocked,
                // Make GenMC print all executions.
                "=all" => ExecutiongraphPrinting::ExploredAndBlocked,
                _ =>
                    return Err(format!(
                        "Invalid suffix to GenMC argument '-Zmiri-genmc-print-exec-graphs', expected '', '=none', '=explored', '=blocked' or '=all'"
                    )),
            }
        } else if trimmed_arg == "estimate" {
            // FIXME(genmc): should this be on by default (like for GenMC)?
            // Enable estimating the execution space and require time before running the actual verification.
            genmc_config.do_estimation = true;
        } else if let Some(estimation_max_str) = trimmed_arg.strip_prefix("estimation-max=") {
            // Set the maximum number of executions to explore during estimation.
            genmc_config.params.estimation_max = estimation_max_str.parse().ok().filter(|estimation_max| *estimation_max > 0).ok_or_else(|| {
                format!(
                    "'-Zmiri-genmc-estimation-max=...' expects a positive integer argument, but got '{estimation_max_str}'"
                )
            })?;
        } else if trimmed_arg == "print-genmc-output" {
            genmc_config.print_genmc_output = true;
        } else if trimmed_arg == "verbose" {
            genmc_config.verbose_output = true;
        } else {
            return Err(format!("Invalid GenMC argument: \"-Zmiri-genmc-{trimmed_arg}\""));
        }
        Ok(())
    }

    /// Validate settings for GenMC mode (NOP if GenMC mode disabled).
    ///
    /// Unsupported configurations return an error.
    /// Adjusts Miri settings where required, printing a warnings if the change might be unexpected for the user.
    pub fn validate(miri_config: &mut MiriConfig, tcx: TyCtxt<'_>) -> Result<(), &'static str> {
        let Some(genmc_config) = miri_config.genmc_config.as_mut() else {
            return Ok(());
        };

        // Check for supported target.
        if tcx.data_layout.endian != Endian::Little || tcx.data_layout.pointer_size().bits() != 64 {
            return Err("GenMC only supports 64bit little-endian targets");
        }

        // Check for disallowed configurations.
        if !miri_config.data_race_detector {
            return Err("Cannot disable data race detection in GenMC mode");
        } else if !miri_config.native_lib.is_empty() {
            return Err("native-lib not supported in GenMC mode.");
        } else if miri_config.isolated_op != IsolatedOp::Reject(RejectOpWith::Abort) {
            return Err("Cannot disable isolation in GenMC mode");
        }

        // Adjust settings where needed.
        if !miri_config.weak_memory_emulation {
            genmc_config.params.disable_weak_memory_emulation = true;
        }
        if miri_config.borrow_tracker.is_some() {
            eprintln!(
                "warning: borrow tracking has been disabled, it is not (yet) supported in GenMC mode."
            );
            miri_config.borrow_tracker = None;
        }
        // We enable fixed scheduling so Miri doesn't randomly yield before a terminator, which anyway
        // would be a NOP in GenMC mode.
        miri_config.fixed_scheduling = true;

        Ok(())
    }
}
