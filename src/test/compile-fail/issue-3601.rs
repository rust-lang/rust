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
