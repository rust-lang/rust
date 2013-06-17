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


use astsrv;
use doc;
use time;

#[cfg(test)] use extract;

/// A single operation on the document model
pub struct Pass {
    name: ~str,
    f: @fn(srv: astsrv::Srv, doc: doc::Doc) -> doc::Doc
}

pub fn run_passes(
    srv: astsrv::Srv,
    doc: doc::Doc,
    passes: ~[Pass]
) -> doc::Doc {
    let mut passno = 0;
    do passes.iter().fold(doc) |doc, pass| {
        debug!("pass #%d", passno);
        passno += 1;
        do time(copy pass.name) {
            (pass.f)(srv.clone(), copy doc)
        }
    }
}

#[test]
fn test_run_passes() {
    fn pass1(
        _srv: astsrv::Srv,
        doc: doc::Doc
    ) -> doc::Doc {
        doc::Doc{
            pages: ~[
                doc::CratePage(doc::CrateDoc{
                    topmod: doc::ModDoc{
                        item: doc::ItemDoc {
                            name: doc.cratemod().name() + "two",
                            .. copy doc.cratemod().item
                        },
                        items: ~[],
                        index: None
                    }
                })
            ]
        }
    }
    fn pass2(
        _srv: astsrv::Srv,
        doc: doc::Doc
    ) -> doc::Doc {
        doc::Doc{
            pages: ~[
                doc::CratePage(doc::CrateDoc{
                    topmod: doc::ModDoc{
                        item: doc::ItemDoc {
                            name: doc.cratemod().name() + "three",
                            .. copy doc.cratemod().item
                        },
                        items: ~[],
                        index: None
                    }
                })
            ]
        }
    }
    let source = ~"";
    do astsrv::from_str(source) |srv| {
        let passes = ~[
            Pass {
                name: ~"",
                f: pass1
            },
            Pass {
                name: ~"",
                f: pass2
            }
        ];
        let doc = extract::from_srv(srv.clone(), ~"one");
        let doc = run_passes(srv, doc, passes);
        assert_eq!(doc.cratemod().name(), ~"onetwothree");
    }
}
