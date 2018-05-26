// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(main_reexport)]
#![feature(process_exitcode_placeholder)]
#![feature(termination_trait_lib)]

mod foo {
    use std::process::ExitCode;

    pub fn bar() -> ExitCode {
        return ExitCode::SUCCESS;
    }
}

use foo::bar as main;
