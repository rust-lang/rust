use {
    SyntaxNodeRef,
    algo::generate,
};


#[derive(Debug, Copy, Clone)]
pub enum WalkEvent<'a> {
    Enter(SyntaxNodeRef<'a>),
    Exit(SyntaxNodeRef<'a>),
}

pub fn walk<'a>(root: SyntaxNodeRef<'a>) -> impl Iterator<Item = WalkEvent<'a>> {
    generate(Some(WalkEvent::Enter(root)), move |pos| {
        let next = match *pos {
            WalkEvent::Enter(node) => match node.first_child() {
                Some(child) => WalkEvent::Enter(child),
                None => WalkEvent::Exit(node),
            },
            WalkEvent::Exit(node) => {
                if node == root {
                    return None;
                }
                match node.next_sibling() {
                    Some(sibling) => WalkEvent::Enter(sibling),
                    None => WalkEvent::Exit(node.parent().unwrap()),
                }
            }
        };
        Some(next)
    })
}
