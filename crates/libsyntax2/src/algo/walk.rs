use {
    SyntaxNodeRef,
    algo::generate,
};

pub fn preorder<'a>(root: SyntaxNodeRef<'a>) -> impl Iterator<Item = SyntaxNodeRef<'a>> {
    walk(root).filter_map(|event| match event {
        WalkEvent::Enter(node) => Some(node),
        WalkEvent::Exit(_) => None,
    })
}

#[derive(Debug, Copy, Clone)]
pub enum WalkEvent<'a> {
    Enter(SyntaxNodeRef<'a>),
    Exit(SyntaxNodeRef<'a>),
}

pub fn walk<'a>(root: SyntaxNodeRef<'a>) -> impl Iterator<Item = WalkEvent<'a>> {
    generate(Some(WalkEvent::Enter(root)), |pos| {
        let next = match *pos {
            WalkEvent::Enter(node) => match node.first_child() {
                Some(child) => WalkEvent::Enter(child),
                None => WalkEvent::Exit(node),
            },
            WalkEvent::Exit(node) => {
                match node.next_sibling() {
                    Some(sibling) => WalkEvent::Enter(sibling),
                    None => match node.parent() {
                        Some(node) => WalkEvent::Exit(node),
                        None => return None,
                    },
                }
            }
        };
        Some(next)
    })
}
