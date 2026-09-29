//@check-pass
#![warn(clippy::wildcard_imports)]

macro_rules! glob_import {
    ($p:path) => {
        #[allow(unused_imports)]
        use $p::*;
        #[allow(dead_code)]
        fn __glob_used(_e: Error) {}
    };
}

glob_import!(std::io);
