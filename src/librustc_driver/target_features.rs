// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use rustc::session::Session;
use syntax::symbol::Symbol;
use rustc_trans;

/// Add `target_feature = "..."` cfgs for a variety of platform
/// specific features (SSE, NEON etc.).
///
/// This is performed by checking whether a whitelisted set of
/// features is available on the target machine, by querying LLVM.
pub fn add_configuration(cfg: &mut ast::CrateConfig, sess: &Session) {
    let tf = Symbol::intern("target_feature");

    for feat in rustc_trans::target_features(sess) {
        cfg.insert((tf, Some(feat)));
    }

    if sess.crt_static() {
        cfg.insert((tf, Some(Symbol::intern("crt-static"))));
    }
}
