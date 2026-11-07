//@check-pass
#![warn(clippy::wildcard_imports)]

macro_rules! glob_import {
    ($p:path) => {
        use $p::*;
        fn __glob_used(_e: Error) {}
    };
}

glob_import!(std::io);
