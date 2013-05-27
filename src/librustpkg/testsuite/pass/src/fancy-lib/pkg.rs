// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::run;

pub fn main() {
    use core::libc::consts::os::posix88::{S_IRUSR, S_IWUSR, S_IXUSR};

    let out_path = Path(~"build/fancy_lib");
    if !os::path_exists(&out_path) {
        assert!(os::make_dir(&out_path, (S_IRUSR | S_IWUSR | S_IXUSR) as i32));
    }

    let file = io::file_writer(&out_path.push("generated.rs"),
                               [io::Create]).get();
    file.write_str("pub fn wheeeee() { for [1, 2, 3].each() |_| { assert!(true); } }");

    // now compile the crate itself
    run::process_status("rustc", [~"src/fancy-lib/fancy-lib.rs", ~"--lib", ~"-o",
                        out_path.push(~"fancy_lib").to_str()]);
}