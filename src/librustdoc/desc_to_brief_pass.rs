// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
Pulls a brief description out of a long description.

If the first paragraph of a long description is short enough then it
is interpreted as the brief description.
*/

use core::prelude::*;

use astsrv;
use doc::ItemUtils;
use doc;
use fold::Fold;
use fold;
use pass::Pass;

use core::iterator::IteratorUtil;
use core::str;
use core::util;

pub fn mk_pass() -> Pass {
    Pass {
        name: ~"desc_to_brief",
        f: run
    }
}

pub fn run(
    _srv: astsrv::Srv,
    doc: doc::Doc
) -> doc::Doc {
    let fold = Fold {
        fold_item: fold_item,
        fold_trait: fold_trait,
        fold_impl: fold_impl,
        .. fold::default_any_fold(())
    };
    (fold.fold_doc)(&fold, doc)
}

fn fold_item(fold: &fold::Fold<()>, doc: doc::ItemDoc) -> doc::ItemDoc {
    let doc = fold::default_seq_fold_item(fold, doc);

    doc::ItemDoc {
        brief: extract(copy doc.desc),
        .. doc
    }
}

fn fold_trait(fold: &fold::Fold<()>, doc: doc::TraitDoc) -> doc::TraitDoc {
    let doc =fold::default_seq_fold_trait(fold, doc);

    doc::TraitDoc {
        methods: doc.methods.map(|doc| doc::MethodDoc {
            brief: extract(copy doc.desc),
            .. copy *doc
        }),
        .. doc
    }
}

fn fold_impl(fold: &fold::Fold<()>, doc: doc::ImplDoc) -> doc::ImplDoc {
    let doc =fold::default_seq_fold_impl(fold, doc);

    doc::ImplDoc {
        methods: doc.methods.map(|doc| doc::MethodDoc {
            brief: extract(copy doc.desc),
            .. copy *doc
        }),
        .. doc
    }
}

pub fn extract(desc: Option<~str>) -> Option<~str> {
    if desc.is_none() {
        return None
    }

    parse_desc((copy desc).get())
}

fn parse_desc(desc: ~str) -> Option<~str> {
    static max_brief_len: uint = 120u;

    match first_sentence(copy desc) {
      Some(first_sentence) => {
        if first_sentence.len() <= max_brief_len {
            Some(first_sentence)
        } else {
            None
        }
      }
      None => None
    }
}

fn first_sentence(s: ~str) -> Option<~str> {
    let paras = paragraphs(s);
    if !paras.is_empty() {
        let first_para = paras.head();
        Some(first_sentence_(*first_para).replace("\n", " "))
    } else {
        None
    }
}

fn first_sentence_(s: &str) -> ~str {
    let mut dotcount = 0;
    // The index of the character following a single dot. This allows
    // Things like [0..1) to appear in the brief description
    let idx = s.find(|ch: char| {
        if ch == '.' {
            dotcount += 1;
            false
        } else if dotcount == 1 {
            true
        } else {
            dotcount = 0;
            false
        }
    });
    match idx {
        Some(idx) if idx > 2u => {
            str::to_owned(s.slice(0, idx - 1))
        }
        _ => {
            if s.ends_with(".") {
                str::to_owned(s)
            } else {
                str::to_owned(s)
            }
        }
    }
}

pub fn paragraphs(s: &str) -> ~[~str] {
    let mut lines = ~[];
    for str::each_line_any(s) |line| { lines.push(line.to_owned()); }
    let mut whitespace_lines = 0;
    let mut accum = ~"";
    let paras = do lines.iter().fold(~[]) |paras, line| {
        let mut res = paras;

        if line.is_whitespace() {
            whitespace_lines += 1;
        } else {
            if whitespace_lines > 0 {
                if !accum.is_empty() {
                    let v = util::replace(&mut accum, ~"");
                    res.push(v);
                }
            }

            whitespace_lines = 0;

            accum = if accum.is_empty() {
                copy *line
            } else {
                accum + "\n" + *line
            }
        }

        res
    };

    if !accum.is_empty() {
        paras + [accum]
    } else {
        paras
    }
}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use astsrv;
    use attr_pass;
    use super::{extract, paragraphs, run};
    use doc;
    use extract;

    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            let doc = (attr_pass::mk_pass().f)(srv.clone(), doc);
            run(srv.clone(), doc)
        }
    }

    #[test]
    fn should_promote_desc() {
        let doc = mk_doc(~"#[doc = \"desc\"] mod m { }");
        assert_eq!(doc.cratemod().mods()[0].brief(), Some(~"desc"));
    }

    #[test]
    fn should_promote_trait_method_desc() {
        let doc = mk_doc(~"trait i { #[doc = \"desc\"] fn a(); }");
        assert!(doc.cratemod().traits()[0].methods[0].brief ==
                Some(~"desc"));
    }

    #[test]
    fn should_promote_impl_method_desc() {
        let doc = mk_doc(
            ~"impl int { #[doc = \"desc\"] fn a() { } }");
        assert!(doc.cratemod().impls()[0].methods[0].brief == Some(~"desc"));
    }

    #[test]
    fn test_paragraphs_1() {
        let paras = paragraphs("1\n\n2");
        assert_eq!(paras, ~[~"1", ~"2"]);
    }

    #[test]
    fn test_paragraphs_2() {
        let paras = paragraphs("\n\n1\n1\n\n2\n\n");
        assert_eq!(paras, ~[~"1\n1", ~"2"]);
    }

    #[test]
    fn should_promote_short_descs() {
        let desc = Some(~"desc");
        let brief = extract(copy desc);
        assert_eq!(brief, desc);
    }

    #[test]
    fn should_not_promote_long_descs() {
        let desc = Some(~"Warkworth Castle is a ruined medieval building
in the town of the same name in the English county of Northumberland,
and the town and castle occupy a loop of the River Coquet, less than a mile
from England's north-east coast. When the castle was founded is uncertain,
but traditionally its construction has been ascribed to Prince Henry of
Scotland in the mid 12th century, although it may have been built by
King Henry II of England when he took control of England'snorthern
counties.");
        let brief = extract(desc);
        assert_eq!(brief, None);
    }

    #[test]
    fn should_promote_first_sentence() {
        let desc = Some(~"Warkworth Castle is a ruined medieval building
in the town. of the same name in the English county of Northumberland,
and the town and castle occupy a loop of the River Coquet, less than a mile
from England's north-east coast. When the castle was founded is uncertain,
but traditionally its construction has been ascribed to Prince Henry of
Scotland in the mid 12th century, although it may have been built by
King Henry II of England when he took control of England'snorthern
counties.");
        let brief = extract(desc);
        assert!(brief == Some(
            ~"Warkworth Castle is a ruined medieval building in the town"));
    }

    #[test]
    fn should_not_consider_double_period_to_end_sentence() {
        let desc = Some(~"Warkworth..Castle is a ruined medieval building
in the town. of the same name in the English county of Northumberland,
and the town and castle occupy a loop of the River Coquet, less than a mile
from England's north-east coast. When the castle was founded is uncertain,
but traditionally its construction has been ascribed to Prince Henry of
Scotland in the mid 12th century, although it may have been built by
King Henry II of England when he took control of England'snorthern
counties.");
        let brief = extract(desc);
        assert!(brief == Some(
            ~"Warkworth..Castle is a ruined medieval building in the town"));
    }

    #[test]
    fn should_not_consider_triple_period_to_end_sentence() {
        let desc = Some(~"Warkworth... Castle is a ruined medieval building
in the town. of the same name in the English county of Northumberland,
and the town and castle occupy a loop of the River Coquet, less than a mile
from England's north-east coast. When the castle was founded is uncertain,
but traditionally its construction has been ascribed to Prince Henry of
Scotland in the mid 12th century, although it may have been built by
King Henry II of England when he took control of England'snorthern
counties.");
        let brief = extract(desc);
        assert!(brief == Some(
            ~"Warkworth... Castle is a ruined medieval building in the town"));
    }
}
