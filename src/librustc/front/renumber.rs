// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session::Session;

use syntax::ast;
use syntax::fold;

static STD_VERSION: &'static str = "0.8-pre";

pub fn renumber_crate(_sess: Session, crate: @ast::Crate) -> @ast::Crate {
    let counter = @mut 0;

    let precursor = @fold::AstFoldFns {
        new_id: |_old_id| {
            let new_id = *counter;
            *counter += 1;
            new_id
        },
        ..*fold::default_ast_fold()
    };

    let fold = fold::make_fold(precursor);

    @fold.fold_crate(crate)
}
