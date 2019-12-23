#![feature(test, rustc_private)]

extern crate test;
use crate::test::Bencher;
mod helpers;
use crate::helpers::*;

#[bench]
fn noop(bencher: &mut Bencher) {
    bencher.iter(|| smoke_helper::main())
}

/*
// really slow
#[bench]
fn noop_miri_full(bencher: &mut Bencher) {
    let path = std::env::var("RUST_SYSROOT").expect("env variable `RUST_SYSROOT` not set");
    bencher.iter(|| {
        let mut process = std::process::Command::new("target/release/miri");
        process.arg("benches/smoke_helper.rs")
               .arg("--sysroot").arg(&path);
        let output = process.output().unwrap();
        if !output.status.success() {
            println!("{}", String::from_utf8(output.stdout).unwrap());
            println!("{}", String::from_utf8(output.stderr).unwrap());
            panic!("failed to run miri");
        }
    })
}
*/

#[bench]
fn noop_miri_interpreter(bencher: &mut Bencher) {
    miri_helper::run("smoke_helper", bencher);
}
