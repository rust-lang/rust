pub use self::ffi::*;

impl Default for GenmcParams {
    fn default() -> Self {
        Self {
            print_random_schedule_seed: false,
            do_symmetry_reduction: false,
            // FIXME(GenMC): Add defaults for remaining parameters
        }
    }
}

#[cxx::bridge]
mod ffi {
    /// Parameters that will be given to GenMC for setting up the model checker.
    /// (The fields of this struct are visible to both Rust and C++)
    #[derive(Clone, Debug)]
    struct GenmcParams {
        pub print_random_schedule_seed: bool,
        pub do_symmetry_reduction: bool,
        // FIXME(GenMC): Add remaining parameters.
    }
    unsafe extern "C++" {
        include!("MiriInterface.hpp");

        type MiriGenMCShim;

        fn createGenmcHandle(config: &GenmcParams) -> UniquePtr<MiriGenMCShim>;
    }
}
