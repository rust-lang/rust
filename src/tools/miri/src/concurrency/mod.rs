pub mod cpu_affinity;
pub mod data_race;
mod data_race_handler;
pub mod init_once;
pub mod sync;
pub mod thread;
mod vector_clock;
pub mod weak_memory;

// Import either the real genmc adapter or a dummy module.
// On unsupported platforms, we always include the dummy module, even if the `genmc` feature is enabled.
// FIXME(genmc,macos): Add `target_os = "macos"` once `https://github.com/dtolnay/cxx/issues/1535` is fixed.
#[cfg_attr(
    not(all(
        feature = "genmc",
        target_os = "linux",
        target_pointer_width = "64",
        target_endian = "little"
    )),
    path = "genmc/dummy.rs"
)]
mod genmc;

pub use self::data_race_handler::{AllocDataRaceHandler, GlobalDataRaceHandler};
pub use self::genmc::{GenmcConfig, GenmcCtx};
pub use self::vector_clock::VClock;
