/// Machinery for macro expansion.
///
/// One of the more complicated things about macros is managing the source code
/// that is produced after expansion. See `HirFileId` and `MacroCallId` for how
/// do we do that.
///
/// When the file-management question is resolved, all that is left is a
/// token-tree-to-token-tree transformation plus hygiene. We don't have either of
/// those yet, so all macros are string based at the moment!
use std::sync::Arc;

use ra_syntax::{
    TextRange, TextUnit, SourceFile, AstNode, SyntaxNode, TreeArc, SyntaxNodePtr,
    ast::{self, NameOwner},
};

use crate::{MacroCallId, PersistentHirDatabase};

// Hard-coded defs for now :-(
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroDef {
    Vec,
    QueryGroup,
}

impl MacroDef {
    /// Expands macro call, returning the expansion and offset to be used to
    /// convert ranges between expansion and original source.
    pub fn ast_expand(macro_call: &ast::MacroCall) -> Option<(TextUnit, MacroExpansion)> {
        let (def, input) = MacroDef::from_call(macro_call)?;
        let exp = def.expand(input)?;
        let off = macro_call.token_tree()?.syntax().range().start();
        Some((off, exp))
    }

    fn from_call(macro_call: &ast::MacroCall) -> Option<(MacroDef, MacroInput)> {
        let def = {
            let path = macro_call.path()?;
            let name_ref = path.segment()?.name_ref()?;
            if name_ref.text() == "vec" {
                MacroDef::Vec
            } else if name_ref.text() == "query_group" {
                MacroDef::QueryGroup
            } else {
                return None;
            }
        };

        let input = {
            let arg = macro_call.token_tree()?.syntax();
            MacroInput { text: arg.text().to_string() }
        };
        Some((def, input))
    }

    fn expand(self, input: MacroInput) -> Option<MacroExpansion> {
        match self {
            MacroDef::Vec => self.expand_vec(input),
            MacroDef::QueryGroup => self.expand_query_group(input),
        }
    }
    fn expand_vec(self, input: MacroInput) -> Option<MacroExpansion> {
        let text = format!(r"fn dummy() {{ {}; }}", input.text);
        let file = SourceFile::parse(&text);
        let array_expr = file.syntax().descendants().find_map(ast::ArrayExpr::cast)?;
        let ptr = SyntaxNodePtr::new(array_expr.syntax());
        let src_range = TextRange::offset_len(0.into(), TextUnit::of_str(&input.text));
        let ranges_map = vec![(src_range, array_expr.syntax().range())];
        let res = MacroExpansion { text, ranges_map, ptr };
        Some(res)
    }
    fn expand_query_group(self, input: MacroInput) -> Option<MacroExpansion> {
        let anchor = "trait ";
        let pos = input.text.find(anchor)? + anchor.len();
        let trait_name =
            input.text[pos..].chars().take_while(|c| c.is_alphabetic()).collect::<String>();
        if trait_name.is_empty() {
            return None;
        }
        let src_range = TextRange::offset_len((pos as u32).into(), TextUnit::of_str(&trait_name));
        let text = format!(r"trait {} {{ }}", trait_name);
        let file = SourceFile::parse(&text);
        let trait_def = file.syntax().descendants().find_map(ast::TraitDef::cast)?;
        let name = trait_def.name()?;
        let ptr = SyntaxNodePtr::new(trait_def.syntax());
        let ranges_map = vec![(src_range, name.syntax().range())];
        let res = MacroExpansion { text, ranges_map, ptr };
        Some(res)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MacroInput {
    // Should be token trees
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroExpansion {
    /// The result of macro expansion. Should be token tree as well.
    text: String,
    /// Correspondence between ranges in the original source code and ranges in
    /// the macro.
    ranges_map: Vec<(TextRange, TextRange)>,
    /// Implementation detail: internally, a macro is expanded to the whole file,
    /// even if it is an expression. This `ptr` selects the actual expansion from
    /// the expanded file.
    ptr: SyntaxNodePtr,
}

impl MacroExpansion {
    // FIXME: does not really make sense, macro expansion is not neccessary a
    // whole file. See `MacroExpansion::ptr` as well.
    pub(crate) fn file(&self) -> TreeArc<SourceFile> {
        SourceFile::parse(&self.text)
    }

    pub fn syntax(&self) -> TreeArc<SyntaxNode> {
        self.ptr.to_node(&self.file()).to_owned()
    }
    /// Maps range in the source code to the range in the expanded code.
    pub fn map_range_forward(&self, src_range: TextRange) -> Option<TextRange> {
        for (s_range, t_range) in self.ranges_map.iter() {
            if src_range.is_subrange(&s_range) {
                let src_at_zero_range = src_range - src_range.start();
                let src_range_offset = src_range.start() - s_range.start();
                let src_range = src_at_zero_range + src_range_offset + t_range.start();
                return Some(src_range);
            }
        }
        None
    }
    /// Maps range in the expanded code to the range in the source code.
    pub fn map_range_back(&self, tgt_range: TextRange) -> Option<TextRange> {
        for (s_range, t_range) in self.ranges_map.iter() {
            if tgt_range.is_subrange(&t_range) {
                let tgt_at_zero_range = tgt_range - tgt_range.start();
                let tgt_range_offset = tgt_range.start() - t_range.start();
                let src_range = tgt_at_zero_range + tgt_range_offset + s_range.start();
                return Some(src_range);
            }
        }
        None
    }
}

pub(crate) fn expand_macro_invocation(
    db: &impl PersistentHirDatabase,
    invoc: MacroCallId,
) -> Option<Arc<MacroExpansion>> {
    let loc = invoc.loc(db);
    let syntax = db.file_item(loc.source_item_id);
    let macro_call = ast::MacroCall::cast(&syntax).unwrap();

    let (def, input) = MacroDef::from_call(macro_call)?;
    def.expand(input).map(Arc::new)
}
