#![feature(box_patterns)]

struct HTMLImageData {
    image: Option<String>,
}

struct ElementData {
    kind: Box<ElementKind>,
}

enum ElementKind {
    HTMLImageElement(HTMLImageData),
}

enum NodeKind {
    Element(ElementData),
}

struct NodeData {
    kind: Box<NodeKind>,
}

fn main() {
    let mut id = HTMLImageData { image: None };
    let ed = ElementData { kind: Box::new(ElementKind::HTMLImageElement(id)) };
    let n = NodeData { kind: Box::new(NodeKind::Element(ed)) };

    // n.b. span could be better
    match n.kind {
        box NodeKind::Element(ed) => match ed.kind {
            //~^ ERROR non-exhaustive patterns
            //~| NOTE the matched value is of type
            //~| NOTE match arms with guards don't count towards exhaustivity
            //~| NOTE pattern `box _` not covered
            //~| NOTE `Box<ElementKind>` defined here
            box ElementKind::HTMLImageElement(ref d) if d.image.is_some() => true,
        },
    };
}
