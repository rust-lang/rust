#![doc = include_str!("../README.md")]

use std::collections::HashSet;
use std::env;
use std::path::PathBuf;

use build_manifest::{Builder, Checksums, Versions};

fn main() {
    let num_threads = if let Some(num) = env::var_os("BUILD_MANIFEST_NUM_THREADS") {
        num.to_str().unwrap().parse().expect("invalid number for BUILD_MANIFEST_NUM_THREADS")
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .expect("failed to initialize Rayon");

    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap());
    let output = PathBuf::from(args.next().unwrap());
    let date = args.next().unwrap();
    let s3_address = args.next().unwrap();
    let channel = args.next().unwrap();

    Builder {
        versions: Versions::new(&channel, &input).unwrap(),
        checksums: build_manifest::t!(Checksums::new()),
        shipped_files: HashSet::new(),

        input,
        output,
        s3_address,
        date,
    }
    .build();
}
