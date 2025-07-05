pub mod cpu_affinity;
pub mod data_race;
mod data_race_handler;
pub mod init_once;
mod range_object_map;
pub mod sync;
pub mod thread;
mod vector_clock;
pub mod weak_memory;

// Import either the real genmc adapter or a dummy module.
#[cfg_attr(not(feature = "genmc"), path = "genmc/dummy.rs")]
mod genmc;

pub use self::data_race_handler::{AllocDataRaceHandler, GlobalDataRaceHandler};
pub use self::genmc::{GenmcConfig, GenmcCtx};
pub use self::vector_clock::VClock;
