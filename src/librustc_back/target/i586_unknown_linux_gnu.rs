// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use target::TargetResult;

pub fn target() -> TargetResult {
    let mut base = super::i686_unknown_linux_gnu::target()?;
    base.options.cpu = "pentium".to_string();
    base.llvm_target = "i586-unknown-linux-gnu".to_string();
    Ok(base)
}
