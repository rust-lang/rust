use std::sync::Arc;

use ra_db::{LocalSyntaxPtr, LocationIntener};
use ra_syntax::{
    TextRange, TextUnit, SourceFileNode, AstNode, SyntaxNode,
    ast,
};

use crate::{SourceRootId, module::ModuleId, SourceItemId, HirDatabase};

/// Def's are a core concept of hir. A `Def` is an Item (function, module, etc)
/// in a specific module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroInvocationId(u32);
ra_db::impl_numeric_id!(MacroInvocationId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MacroInvocationLoc {
    source_root_id: SourceRootId,
    module_id: ModuleId,
    source_item_id: SourceItemId,
}

impl MacroInvocationId {
    pub(crate) fn loc(
        self,
        db: &impl AsRef<LocationIntener<MacroInvocationLoc, MacroInvocationId>>,
    ) -> MacroInvocationLoc {
        db.as_ref().id2loc(self)
    }
}

impl MacroInvocationLoc {
    #[allow(unused)]
    pub(crate) fn id(
        &self,
        db: &impl AsRef<LocationIntener<MacroInvocationLoc, MacroInvocationId>>,
    ) -> MacroInvocationId {
        db.as_ref().loc2id(&self)
    }
}

// Hard-coded defs for now :-(
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MacroDef {
    CTry,
}

impl MacroDef {
    pub fn ast_expand(macro_call: ast::MacroCall) -> Option<(TextUnit, MacroExpansion)> {
        let (def, input) = MacroDef::from_call(macro_call)?;
        let exp = def.expand(input)?;
        let off = macro_call.token_tree()?.syntax().range().start();
        Some((off, exp))
    }

    fn from_call(macro_call: ast::MacroCall) -> Option<(MacroDef, MacroInput)> {
        let def = {
            let path = macro_call.path()?;
            if path.qualifier().is_some() {
                return None;
            }
            let name_ref = path.segment()?.name_ref()?;
            if name_ref.text() != "ctry" {
                return None;
            }
            MacroDef::CTry
        };

        let input = {
            let arg = macro_call.token_tree()?.syntax();
            MacroInput {
                text: arg.text().to_string(),
            }
        };
        Some((def, input))
    }

    fn expand(self, input: MacroInput) -> Option<MacroExpansion> {
        let MacroDef::CTry = self;
        let text = format!(
            r"
                fn dummy() {{
                    match {} {{
                        None => return Ok(None),
                        Some(it) => it,
                    }}
                }}",
            input.text
        );
        let file = SourceFileNode::parse(&text);
        let match_expr = file.syntax().descendants().find_map(ast::MatchExpr::cast)?;
        let match_arg = match_expr.expr()?;
        let ptr = LocalSyntaxPtr::new(match_arg.syntax());
        let src_range = TextRange::offset_len(0.into(), TextUnit::of_str(&input.text));
        let ranges_map = vec![(src_range, match_arg.syntax().range())];
        let res = MacroExpansion {
            text,
            ranges_map,
            ptr,
        };
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
    text: String,
    ranges_map: Vec<(TextRange, TextRange)>,
    ptr: LocalSyntaxPtr,
}

impl MacroExpansion {
    pub fn file(&self) -> SourceFileNode {
        SourceFileNode::parse(&self.text)
    }

    pub fn syntax(&self) -> SyntaxNode {
        self.ptr.resolve(&self.file())
    }
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
}

pub(crate) fn expand_macro_invocation(
    db: &impl HirDatabase,
    invoc: MacroInvocationId,
) -> Option<Arc<MacroExpansion>> {
    let loc = invoc.loc(db);
    let syntax = db.file_item(loc.source_item_id);
    let syntax = syntax.borrowed();
    let macro_call = ast::MacroCall::cast(syntax).unwrap();

    let (def, input) = MacroDef::from_call(macro_call)?;
    def.expand(input).map(Arc::new)
}
