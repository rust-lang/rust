#!/usr/bin/env bash
#![allow()] /*This line is ignored by bash
# This block is ignored by rustc
set -e
echo "[BUILD] y.rs" 1>&2
rustc $0 -o ${0/.rs/.bin} -g
exec ${0/.rs/.bin} $@
*/

//! The build system for cg_clif
//!
//! # Manual compilation
//!
//! If your system doesn't support shell scripts you can manually compile and run this file using
//! for example:
//!
//! ```shell
//! $ rustc y.rs -o y.bin
//! $ ./y.bin
//! ```
//!
//! # Naming
//!
//! The name `y.rs` was chosen to not conflict with rustc's `x.py`.

#[path = "build_system/mod.rs"]
mod build_system;

fn main() {
    build_system::main();
}
