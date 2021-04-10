//! Proc Macro Expander stub

use crate::db::AstDatabase;
use base_db::{CrateId, ProcMacroId};
use tt::buffer::{Cursor, TokenBuffer};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct ProcMacroExpander {
    krate: CrateId,
    proc_macro_id: Option<ProcMacroId>,
}

macro_rules! err {
    ($fmt:literal, $($tt:tt),*) => {
        mbe::ExpandError::ProcMacroError(tt::ExpansionError::Unknown(format!($fmt, $($tt),*)))
    };
    ($fmt:literal) => {
        mbe::ExpandError::ProcMacroError(tt::ExpansionError::Unknown($fmt.to_string()))
    }
}

impl ProcMacroExpander {
    pub fn new(krate: CrateId, proc_macro_id: ProcMacroId) -> Self {
        Self { krate, proc_macro_id: Some(proc_macro_id) }
    }

    pub fn dummy(krate: CrateId) -> Self {
        // FIXME: Should store the name for better errors
        Self { krate, proc_macro_id: None }
    }

    pub fn expand(
        self,
        db: &dyn AstDatabase,
        calling_crate: CrateId,
        tt: &tt::Subtree,
    ) -> Result<tt::Subtree, mbe::ExpandError> {
        match self.proc_macro_id {
            Some(id) => {
                let krate_graph = db.crate_graph();
                let proc_macro = krate_graph[self.krate]
                    .proc_macro
                    .get(id.0 as usize)
                    .clone()
                    .ok_or_else(|| err!("No derive macro found."))?;

                let tt = remove_derive_attrs(tt)
                    .ok_or_else(|| err!("Fail to remove derive for custom derive"))?;

                // Proc macros have access to the environment variables of the invoking crate.
                let env = &krate_graph[calling_crate].env;

                proc_macro.expander.expand(&tt, None, &env).map_err(mbe::ExpandError::from)
            }
            None => Err(mbe::ExpandError::UnresolvedProcMacro),
        }
    }
}

fn eat_punct(cursor: &mut Cursor, c: char) -> bool {
    if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Punct(punct), _)) = cursor.token_tree() {
        if punct.char == c {
            *cursor = cursor.bump();
            return true;
        }
    }
    false
}

fn eat_subtree(cursor: &mut Cursor, kind: tt::DelimiterKind) -> bool {
    if let Some(tt::buffer::TokenTreeRef::Subtree(subtree, _)) = cursor.token_tree() {
        if Some(kind) == subtree.delimiter_kind() {
            *cursor = cursor.bump_subtree();
            return true;
        }
    }
    false
}

fn eat_ident(cursor: &mut Cursor, t: &str) -> bool {
    if let Some(tt::buffer::TokenTreeRef::Leaf(tt::Leaf::Ident(ident), _)) = cursor.token_tree() {
        if t == ident.text.as_str() {
            *cursor = cursor.bump();
            return true;
        }
    }
    false
}

fn remove_derive_attrs(tt: &tt::Subtree) -> Option<tt::Subtree> {
    let buffer = TokenBuffer::from_tokens(&tt.token_trees);
    let mut p = buffer.begin();
    let mut result = tt::Subtree::default();

    while !p.eof() {
        let curr = p;

        if eat_punct(&mut p, '#') {
            eat_punct(&mut p, '!');
            let parent = p;
            if eat_subtree(&mut p, tt::DelimiterKind::Bracket) {
                if eat_ident(&mut p, "derive") {
                    p = parent.bump();
                    continue;
                }
            }
        }

        result.token_trees.push(curr.token_tree()?.cloned());
        p = curr.bump();
    }

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_utils::assert_eq_text;

    #[test]
    fn test_remove_derive_attrs() {
        let tt = mbe::parse_to_token_tree(
            r#"
    #[allow(unused)]
    #[derive(Copy)]
    #[derive(Hello)]
    struct A {
        bar: u32
    }
"#,
        )
        .unwrap()
        .0;
        let result = format!("{:#?}", remove_derive_attrs(&tt).unwrap());

        assert_eq_text!(
            r#"
SUBTREE $
  PUNCH   # [alone] 0
  SUBTREE [] 1
    IDENT   allow 2
    SUBTREE () 3
      IDENT   unused 4
  IDENT   struct 15
  IDENT   A 16
  SUBTREE {} 17
    IDENT   bar 18
    PUNCH   : [alone] 19
    IDENT   u32 20
"#
            .trim(),
            &result
        );
    }
}
