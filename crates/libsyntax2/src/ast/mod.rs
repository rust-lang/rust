mod generated;

use itertools::Itertools;
use smol_str::SmolStr;

use {
    SyntaxNode, SyntaxNodeRef, OwnedRoot, TreeRoot, SyntaxError,
    SyntaxKind::*,
};
pub use self::generated::*;

pub trait AstNode<R: TreeRoot> {
    fn cast(syntax: SyntaxNode<R>) -> Option<Self>
        where Self: Sized;
    fn syntax(&self) -> &SyntaxNode<R>;
    fn syntax_ref<'a>(&'a self) -> SyntaxNodeRef<'a> where R: 'a {
        self.syntax().as_ref()
    }
}

pub trait NameOwner<R: TreeRoot>: AstNode<R> {
    fn name(&self) -> Option<Name<R>> {
        self.syntax()
            .children()
            .filter_map(Name::cast)
            .next()
    }
}

pub trait AttrsOwner<R: TreeRoot>: AstNode<R> {
    fn attrs<'a>(&'a self) -> Box<Iterator<Item=Attr<R>> + 'a> where R: 'a {
        let it = self.syntax().children()
            .filter_map(Attr::cast);
        Box::new(it)
    }
}

impl File<OwnedRoot> {
    pub fn parse(text: &str) -> Self {
        File::cast(::parse(text)).unwrap()
    }
}

impl<R: TreeRoot> File<R> {
    pub fn errors(&self) -> Vec<SyntaxError> {
        self.syntax().root.syntax_root().errors.clone()
    }
}

impl<R: TreeRoot> FnDef<R> {
    pub fn has_atom_attr(&self, atom: &str) -> bool {
        self.attrs()
            .filter_map(|x| x.as_atom())
            .any(|x| x == atom)
    }
}

impl<R: TreeRoot> Attr<R> {
    pub fn as_atom(&self) -> Option<SmolStr> {
        let tt = self.value()?;
        let (_bra, attr, _ket) = tt.syntax().children().collect_tuple()?;
        if attr.kind() == IDENT {
            Some(attr.leaf_text().unwrap())
        } else {
            None
        }
    }

    pub fn as_call(&self) -> Option<(SmolStr, TokenTree<R>)> {
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

impl<R: TreeRoot> Name<R> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child()
            .unwrap();
        ident.leaf_text().unwrap()
    }
}

impl<R: TreeRoot> NameRef<R> {
    pub fn text(&self) -> SmolStr {
        let ident = self.syntax().first_child()
            .unwrap();
        ident.leaf_text().unwrap()
    }
}

impl <R: TreeRoot> ImplItem<R> {
    pub fn target_type(&self) -> Option<TypeRef<R>> {
        match self.target() {
            (Some(t), None) | (_, Some(t)) => Some(t),
            _ => None,
        }
    }

    pub fn target_trait(&self) -> Option<TypeRef<R>> {
        match self.target() {
            (Some(t), Some(_)) => Some(t),
            _ => None,
        }
    }

    fn target(&self) -> (Option<TypeRef<R>>, Option<TypeRef<R>>) {
        let mut types = self.syntax().children().filter_map(TypeRef::cast);
        let first = types.next();
        let second = types.next();
        (first, second)
    }
}

impl <R: TreeRoot> Module<R> {
    pub fn has_semi(&self) -> bool {
        match self.syntax_ref().last_child() {
            None => false,
            Some(node) => node.kind() == SEMI,
        }
    }
}
