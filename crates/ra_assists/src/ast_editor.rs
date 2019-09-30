use std::{iter, ops::RangeInclusive};

use ra_syntax::{
    algo,
    ast::{self, TypeBoundsOwner},
    AstNode, SyntaxElement,
};
use ra_text_edit::TextEditBuilder;
use rustc_hash::FxHashMap;

pub struct AstEditor<N: AstNode> {
    original_ast: N,
    ast: N,
}

impl<N: AstNode> AstEditor<N> {
    pub fn new(node: N) -> AstEditor<N>
    where
        N: Clone,
    {
        AstEditor { original_ast: node.clone(), ast: node }
    }

    pub fn into_text_edit(self, builder: &mut TextEditBuilder) {
        algo::diff(&self.original_ast.syntax(), self.ast().syntax()).into_text_edit(builder)
    }

    pub fn ast(&self) -> &N {
        &self.ast
    }

    pub fn replace_descendants<T: AstNode>(
        &mut self,
        replacement_map: impl Iterator<Item = (T, T)>,
    ) -> &mut Self {
        let map = replacement_map
            .map(|(from, to)| (from.syntax().clone().into(), to.syntax().clone().into()))
            .collect::<FxHashMap<_, _>>();
        let new_syntax = algo::replace_descendants(self.ast.syntax(), &map);
        self.ast = N::cast(new_syntax).unwrap();
        self
    }

    #[must_use]
    fn replace_children(
        &self,
        to_delete: RangeInclusive<SyntaxElement>,
        mut to_insert: impl Iterator<Item = SyntaxElement>,
    ) -> N {
        let new_syntax = algo::replace_children(self.ast().syntax(), to_delete, &mut to_insert);
        N::cast(new_syntax).unwrap()
    }
}
