// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::vec;
use driver::session::Session;
use syntax::parse;
use syntax::ast;
use syntax::codemap::spanned;

pub fn inject_intrinsic(sess: Session, crate: @ast::crate) -> @ast::crate {
    let intrinsic_module = @(include_str!("intrinsic.rs").to_owned());

    let item = parse::parse_item_from_source_str(~"<intrinsic>",
                                                 intrinsic_module,
                                                 /*bad*/copy sess.opts.cfg,
                                                 ~[],
                                                 sess.parse_sess);
    let item =
        match item {
          Some(i) => i,
          None => {
            sess.fatal("no item found in intrinsic module");
          }
        };

    let items = vec::append(~[item], crate.node.module.items);

    @spanned {
        node: ast::crate_ {
            module: ast::_mod {
                items: items,
                .. /*bad*/copy crate.node.module
            },
            .. /*bad*/copy crate.node
        },
        .. /*bad*/copy *crate
    }
}
