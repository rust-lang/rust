use SyntaxNodeRef;

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
    let mut done = false;
    ::itertools::unfold(WalkEvent::Enter(root), move |pos| {
        if done {
            return None;
        }
        let res = *pos;
        *pos = match *pos {
            WalkEvent::Enter(node) => match node.first_child() {
                Some(child) => WalkEvent::Enter(child),
                None => WalkEvent::Exit(node),
            },
            WalkEvent::Exit(node) => {
                if node == root {
                    done = true;
                    WalkEvent::Exit(node)
                } else {
                    match node.next_sibling() {
                        Some(sibling) => WalkEvent::Enter(sibling),
                        None => match node.parent() {
                            Some(node) => WalkEvent::Exit(node),
                            None => WalkEvent::Exit(node),
                        },
                    }
                }
            }
        };
        Some(res)
    })
}
