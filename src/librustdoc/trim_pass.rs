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

use pass::Pass;
use text_pass;

pub fn mk_pass() -> Pass {
    text_pass::mk_pass(~"trim", |s| s.trim().to_owned() )
}

#[cfg(test)]
mod test {
    use astsrv;
    use attr_pass;
    use doc;
    use extract;
    use trim_pass::mk_pass;

    fn mk_doc(source: ~str) -> doc::Doc {
        do astsrv::from_str(copy source) |srv| {
            let doc = extract::from_srv(srv.clone(), ~"");
            let doc = (attr_pass::mk_pass().f)(srv.clone(), doc);
            (mk_pass().f)(srv.clone(), doc)
        }
    }

    #[test]
    fn should_trim_text() {
        use core::option::Some;

        let doc = mk_doc(~"#[doc = \" desc \"] \
                                 mod m {
}");
        assert!(doc.cratemod().mods()[0].desc() == Some(~"desc"));
    }
}
