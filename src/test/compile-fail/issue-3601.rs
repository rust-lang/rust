// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct HTMLImageData {
    image: Option<~str>
}

struct ElementData {
    kind: Box<ElementKind>
}

enum ElementKind {
    HTMLImageElement(HTMLImageData)
}

enum NodeKind {
    Element(ElementData)
}

struct NodeData {
    kind: Box<NodeKind>,
}

fn main() {
    let mut id = HTMLImageData { image: None };
    let ed = ElementData { kind: box HTMLImageElement(id) };
    let n = NodeData {kind : box Element(ed)};
    // n.b. span could be better
    match n.kind {
        box Element(ed) => match ed.kind { //~ ERROR non-exhaustive patterns
            box HTMLImageElement(ref d) if d.image.is_some() => { true }
        },
        _ => fail!("WAT") //~ ERROR unreachable pattern
    };
}
