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
use extract;
use time;

use core::vec;

/// A single operation on the document model
pub type Pass = {
    name: ~str,
    f: fn~(srv: astsrv::Srv, +doc: doc::Doc) -> doc::Doc
};

pub fn run_passes(
    srv: astsrv::Srv,
    +doc: doc::Doc,
    +passes: ~[Pass]
) -> doc::Doc {
    let mut passno = 0;
    do vec::foldl(doc, passes) |doc, pass| {
        log(debug, fmt!("pass #%d", passno));
        passno += 1;
        log(debug, doc);
        do time(pass.name) {
            (pass.f)(srv, doc)
        }
    }
}

#[test]
fn test_run_passes() {
    fn pass1(
        _srv: astsrv::Srv,
        +doc: doc::Doc
    ) -> doc::Doc {
        doc::Doc_({
            pages: ~[
                doc::CratePage({
                    topmod: doc::ModDoc_({
                        item: {
                            name: doc.cratemod().name() + ~"two",
                            .. doc.cratemod().item
                        },
                        items: ~[],
                        index: None
                    })
                })
            ]
        })
    }
    fn pass2(
        _srv: astsrv::Srv,
        +doc: doc::Doc
    ) -> doc::Doc {
        doc::Doc_({
            pages: ~[
                doc::CratePage({
                    topmod: doc::ModDoc_({
                        item: {
                            name: doc.cratemod().name() + ~"three",
                            .. doc.cratemod().item
                        },
                        items: ~[],
                        index: None
                    })
                })
            ]
        })
    }
    let source = ~"";
    do astsrv::from_str(source) |srv| {
        let passes = ~[
            {
                name: ~"",
                f: pass1
            },
            {
                name: ~"",
                f: pass2
            }
        ];
        let doc = extract::from_srv(srv, ~"one");
        let doc = run_passes(srv, doc, passes);
        assert doc.cratemod().name() == ~"onetwothree";
    }
}
