// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
struct HTMLImageData {
    mut image: Option<~str>
}

struct ElementData {
    kind: ~ElementKind
}

enum ElementKind {
    HTMLImageElement(HTMLImageData)
}

enum NodeKind {
    Element(ElementData)
}

enum NodeData = {
    kind: ~NodeKind
};

fn main() {
    let id = HTMLImageData { image: None };
    let ed = ElementData { kind: ~HTMLImageElement(id) };
    let n = NodeData({kind : ~Element(ed)});
    match n.kind {
        ~Element(ed) => match ed.kind {
            ~HTMLImageElement(d) if d.image.is_some() => { true }
        },
        _ => fail ~"WAT" //~ ERROR wat
    };
}
