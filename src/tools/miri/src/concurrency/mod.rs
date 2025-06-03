pub mod cpu_affinity;
pub mod data_race;
mod data_race_handler;
pub mod init_once;
mod range_object_map;
pub mod sync;
pub mod thread;
mod vector_clock;
pub mod weak_memory;

// cfg(bootstrap)
macro_rules! cfg_select_dispatch {
    ($($tokens:tt)*) => {
        #[cfg(bootstrap)]
        cfg_match! { $($tokens)* }

        #[cfg(not(bootstrap))]
        cfg_select! { $($tokens)* }
    };
}

// Import either the real genmc adapter or a dummy module.
cfg_select_dispatch! {
    feature = "genmc" => {
        mod genmc;
        pub use self::genmc::{GenmcCtx, GenmcConfig};
    }
    _ => {
        #[path = "genmc/dummy.rs"]
        mod genmc_dummy;
        use self::genmc_dummy as genmc;
        pub use self::genmc::{GenmcCtx, GenmcConfig};
    }
}

pub use self::data_race_handler::{AllocDataRaceHandler, GlobalDataRaceHandler};
pub use self::vector_clock::VClock;
