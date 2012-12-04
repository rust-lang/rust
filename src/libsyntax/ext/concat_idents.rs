// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use base::*;

fn expand_syntax_ext(cx: ext_ctxt, sp: codemap::span, arg: ast::mac_arg,
                     _body: ast::mac_body) -> @ast::expr {
    let args = get_mac_args_no_max(cx,sp,arg,1u,~"concat_idents");
    let mut res_str = ~"";
    for args.each |e| {
        res_str += *cx.parse_sess().interner.get(
            expr_to_ident(cx, *e, ~"expected an ident"));
    }
    let res = cx.parse_sess().interner.intern(@res_str);

    return @{id: cx.next_id(),
          callee_id: cx.next_id(),
          node: ast::expr_path(@{span: sp, global: false, idents: ~[res],
                                 rp: None, types: ~[]}),
          span: sp};
}
