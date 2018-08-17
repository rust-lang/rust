mod generated;

use itertools::Itertools;
use smol_str::SmolStr;

use {
    SyntaxNode, SyntaxNodeRef, TreeRoot, SyntaxError,
    SyntaxKind::*,
};
pub use self::generated::*;

pub trait AstNode<'a>: Clone + Copy {
    fn cast(syntax: SyntaxNodeRef<'a>) -> Option<Self>
        where Self: Sized;
    fn syntax(self) -> SyntaxNodeRef<'a>;
}

pub trait NameOwner<'a>: AstNode<'a> {
    fn name(self) -> Option<Name<'a>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

pub trait AttrsOwner<'a>: AstNode<'a> {
    fn attrs(&self) -> Box<Iterator<Item=Attr<'a>> + 'a> {
        let it = self.syntax().children()
            .filter_map(Attr::cast);
        Box::new(it)
    }
}

#[derive(Clone, Debug)]
pub struct ParsedFile {
    root: SyntaxNode
}

impl ParsedFile {
    pub fn parse(text: &str) -> Self {
        let root = ::parse(text);
        ParsedFile { root }
    }
    pub fn ast(&self) -> File {
        File::cast(self.syntax()).unwrap()
    }
    pub fn syntax(&self) -> SyntaxNodeRef {
        self.root.as_ref()
    }
    pub fn errors(&self) -> Vec<SyntaxError> {
        self.syntax().root.syntax_root().errors.clone()
    }

}

impl<'a> FnDef<'a> {
    pub fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs()
            .filter_map(|x| x.as_atom())
            .any(|x| x == atom)
    }
}

impl<'a> Attr<'a> {
    pub fn as_atom(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let (_bra, attr, _ket) = tt.syntax().children().collect_tuple()?;
        if attr.kind() == IDENT {
            Some(attr.leaf_text().unwrap())
        } else {
            None
        }
    }

    pub fn as_call(&self) -> Option<(SmolStr, TokenTree<'a>)> {
        let tt = self.value()?;
        let (_bra, attr, args, _ket) = tt.syntax().children().collect_tuple()?;
        let args = TokenTree::cast(args)?;
        if attr.kind() == IDENT {
            Some((attr.leaf_text().unwrap(), args))
        } else {
            None
        }
    }
}

impl<'a> Name<'a> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child()
            .unwrap();
        ident.leaf_text().unwrap()
    }
}

impl<'a> NameRef<'a> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child()
            .unwrap();
        ident.leaf_text().unwrap()
    }
}

impl<'a> ImplItem<'a> {
    pub fn target_type(&self) -> Option<TypeRef<'a>> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn target_trait(&self) -> Option<TypeRef<'a>> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(&self) -> (Option<TypeRef<'a>>, Option<TypeRef<'a>>) {
        let mut types = self.syntax().children().filter_map(TypeRef::cast);
        let first = types.next();
        let second = types.next();
        (first, second)
    }
}

impl<'a> Module<'a> {
    pub fn has_semi(&self) -> bool {
        match self.syntax().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}
