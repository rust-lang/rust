#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NodeKind(u16);

pub struct File {
    text: String,
    nodes: Vec<NodeData>,
}

struct NodeData {
    kind: NodeKind,
    range: (u32, u32),
    parent: Option<u32>,
    first_child: Option<u32>,
    next_sibling: Option<u32>,
}

#[derive(Clone, Copy)]
pub struct Node<'f> {
    file: &'f File,
    idx: u32,
}

pub struct Children<'f> {
    next: Option<Node<'f>>,
}

impl File {
    pub fn root<'f>(&'f self) -> Node<'f> {
        assert!(!self.nodes.is_empty());
        Node { file: self, idx: 0 }
    }
}

impl<'f> Node<'f> {
    pub fn kind(&self) -> NodeKind {
        self.data().kind
    }

    pub fn text(&self) -> &'f str {
        let (start, end) = self.data().range;
        &self.file.text[start as usize..end as usize]
    }

    pub fn parent(&self) -> Option<Node<'f>> {
        self.as_node(self.data().parent)
    }

    pub fn children(&self) -> Children<'f> {
        Children { next: self.as_node(self.data().first_child) }
    }

    fn data(&self) -> &'f NodeData {
        &self.file.nodes[self.idx as usize]
    }

    fn as_node(&self, idx: Option<u32>) -> Option<Node<'f>> {
        idx.map(|idx| Node { file: self.file, idx })
    }
}

impl<'f> Iterator for Children<'f> {
    type Item = Node<'f>;

    fn next(&mut self) -> Option<Node<'f>> {
        let next = self.next;
        self.next = next.and_then(|node| node.as_node(node.data().next_sibling));
        next
    }
}

pub const ERROR: NodeKind = NodeKind(0);
pub const WHITESPACE: NodeKind = NodeKind(1);
pub const STRUCT_KW: NodeKind = NodeKind(2);
pub const IDENT: NodeKind = NodeKind(3);
pub const L_CURLY: NodeKind = NodeKind(4);
pub const R_CURLY: NodeKind = NodeKind(5);
pub const COLON: NodeKind = NodeKind(6);
pub const COMMA: NodeKind = NodeKind(7);
pub const AMP: NodeKind = NodeKind(8);
pub const LINE_COMMENT: NodeKind = NodeKind(9);
pub const FILE: NodeKind = NodeKind(10);
pub const STRUCT_DEF: NodeKind = NodeKind(11);
pub const FIELD_DEF: NodeKind = NodeKind(12);
pub const TYPE_REF: NodeKind = NodeKind(13);


pub trait AstNode<'f>: Copy + 'f {
    fn new(node: Node<'f>) -> Option<Self>;
    fn node(&self) -> Node<'f>;
}

pub fn child_of_kind<'f>(node: Node<'f>, kind: NodeKind) -> Option<Node<'f>> {
    node.children().find(|child| child.kind() == kind)
}

pub fn ast_children<'f, A: AstNode<'f>>(node: Node<'f>) -> Box<Iterator<Item=A> + 'f> {
    Box::new(node.children().filter_map(A::new))
}

#[derive(Clone, Copy)]
pub struct StructDef<'f>(Node<'f>);

#[derive(Clone, Copy)]
pub struct FieldDef<'f>(Node<'f>);

#[derive(Clone, Copy)]
pub struct TypeRef<'f>(Node<'f>);

pub trait NameOwner<'f>: AstNode<'f> {
    fn name_ident(&self) -> Node<'f> {
        child_of_kind(self.node(), IDENT).unwrap()
    }

    fn name(&self) -> &'f str { self.name_ident().text() }
}


impl<'f> AstNode<'f> for StructDef<'f> {
    fn new(node: Node<'f>) -> Option<Self> {
        if node.kind() == STRUCT_DEF { Some(StructDef(node)) } else { None }
    }
    fn node(&self) -> Node<'f> { self.0 }
}

impl<'f> AstNode<'f> for FieldDef<'f> {
    fn new(node: Node<'f>) -> Option<Self> {
        if node.kind() == FIELD_DEF { Some(FieldDef(node)) } else { None }
    }
    fn node(&self) -> Node<'f> { self.0 }
}

impl<'f> AstNode<'f> for TypeRef<'f> {
    fn new(node: Node<'f>) -> Option<Self> {
        if node.kind() == TYPE_REF { Some(TypeRef(node)) } else { None }
    }
    fn node(&self) -> Node<'f> { self.0 }
}

impl<'f> NameOwner<'f> for StructDef<'f> {}
impl<'f> NameOwner<'f> for FieldDef<'f> {}

impl<'f> StructDef<'f> {
    pub fn fields(&self) -> Box<Iterator<Item=FieldDef<'f>> + 'f> {
        ast_children(self.node())
    }
}

impl<'f> FieldDef<'f> {
    pub fn type_ref(&self) -> Option<TypeRef<'f>> {
        ast_children(self.node()).next()
    }
}