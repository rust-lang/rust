//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use std::sync::Arc;

use base_db::CrateId;
use either::Either;
use mbe::Origin;
use parser::SyntaxKind;
use syntax::{ast, AstNode, SyntaxNode, TextRange, TextSize};

use crate::{
    db::{self, AstDatabase},
    name::{AsName, Name},
    HirFileId, HirFileIdRepr, InFile, MacroCallId, MacroCallLoc, MacroDefKind, MacroFile,
};

#[derive(Clone, Debug)]
pub struct Hygiene {
    frames: Option<HygieneFrames>,
}

impl Hygiene {
    pub fn new(db: &dyn AstDatabase, file_id: HirFileId) -> Hygiene {
        Hygiene { frames: Some(HygieneFrames::new(db, file_id)) }
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene { frames: None }
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(&self, name_ref: ast::NameRef) -> Either<Name, CrateId> {
        if let Some(frames) = &self.frames {
            if name_ref.text() == "$crate" {
                if let Some(krate) = frames.root_crate(name_ref.syntax()) {
                    return Either::Right(krate);
                }
            }
        }

        Either::Left(name_ref.as_name())
    }

    pub fn local_inner_macros(&self, path: ast::Path) -> Option<CrateId> {
        let mut token = path.syntax().first_token()?.text_range();
        let frames = self.frames.as_ref()?;
        let mut current = frames.0.clone();

        loop {
            let (mapped, origin) = current.expansion.as_ref()?.map_ident_up(token)?;
            if origin == Origin::Def {
                return if current.local_inner { frames.root_crate(path.syntax()) } else { None };
            }
            current = current.call_site.as_ref()?.clone();
            token = mapped.value;
        }
    }
}

#[derive(Clone, Debug)]
struct HygieneFrames(Arc<HygieneFrame>);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HygieneFrame {
    expansion: Option<HygieneInfo>,

    // Indicate this is a local inner macro
    local_inner: bool,
    krate: Option<CrateId>,

    call_site: Option<Arc<HygieneFrame>>,
    def_site: Option<Arc<HygieneFrame>>,
}

impl HygieneFrames {
    fn new(db: &dyn AstDatabase, file_id: HirFileId) -> Self {
        // Note that this intentionally avoids the `hygiene_frame` query to avoid blowing up memory
        // usage. The query is only helpful for nested `HygieneFrame`s as it avoids redundant work.
        HygieneFrames(Arc::new(HygieneFrame::new(db, file_id)))
    }

    fn root_crate(&self, node: &SyntaxNode) -> Option<CrateId> {
        let mut token = node.first_token()?.text_range();
        let mut result = self.0.krate;
        let mut current = self.0.clone();

        while let Some((mapped, origin)) =
            current.expansion.as_ref().and_then(|it| it.map_ident_up(token))
        {
            result = current.krate;

            let site = match origin {
                Origin::Def => &current.def_site,
                Origin::Call => &current.call_site,
            };

            let site = match site {
                None => break,
                Some(it) => it,
            };

            current = site.clone();
            token = mapped.value;
        }

        result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HygieneInfo {
    arg_start: InFile<TextSize>,
    /// The `macro_rules!` arguments.
    def_start: Option<InFile<TextSize>>,

    macro_def: Arc<(db::TokenExpander, mbe::TokenMap)>,
    macro_arg: Arc<(tt::Subtree, mbe::TokenMap)>,
    exp_map: Arc<mbe::TokenMap>,
}

impl HygieneInfo {
    fn map_ident_up(&self, token: TextRange) -> Option<(InFile<TextRange>, Origin)> {
        let token_id = self.exp_map.token_by_range(token)?;

        let (token_id, origin) = self.macro_def.0.map_id_up(token_id);
        let (token_map, tt) = match origin {
            mbe::Origin::Call => (&self.macro_arg.1, self.arg_start),
            mbe::Origin::Def => (
                &self.macro_def.1,
                *self.def_start.as_ref().expect("`Origin::Def` used with non-`macro_rules!` macro"),
            ),
        };

        let range = token_map.range_by_token(token_id)?.by_kind(SyntaxKind::IDENT)?;
        Some((tt.with_value(range + tt.value), origin))
    }
}

fn make_hygiene_info(
    db: &dyn AstDatabase,
    macro_file: MacroFile,
    loc: &MacroCallLoc,
) -> Option<HygieneInfo> {
    let arg_tt = loc.kind.arg(db)?;

    let def_offset = loc.def.ast_id().left().and_then(|id| {
        let def_tt = match id.to_node(db) {
            ast::Macro::MacroRules(mac) => mac.token_tree()?.syntax().text_range().start(),
            ast::Macro::MacroDef(_) => return None,
        };
        Some(InFile::new(id.file_id, def_tt))
    });

    let macro_def = db.macro_def(loc.def)?;
    let (_, exp_map) = db.parse_macro_expansion(macro_file).value?;
    let macro_arg = db.macro_arg(macro_file.macro_call_id)?;

    Some(HygieneInfo {
        arg_start: InFile::new(loc.kind.file_id(), arg_tt.text_range().start()),
        def_start: def_offset,
        macro_arg,
        macro_def,
        exp_map,
    })
}

impl HygieneFrame {
    pub(crate) fn new(db: &dyn AstDatabase, file_id: HirFileId) -> HygieneFrame {
        let (info, krate, local_inner) = match file_id.0 {
            HirFileIdRepr::FileId(_) => (None, None, false),
            HirFileIdRepr::MacroFile(macro_file) => match macro_file.macro_call_id {
                MacroCallId::EagerMacro(_id) => (None, None, false),
                MacroCallId::LazyMacro(id) => {
                    let loc = db.lookup_intern_macro(id);
                    let info = make_hygiene_info(db, macro_file, &loc);
                    match loc.def.kind {
                        MacroDefKind::Declarative(_) => {
                            (info, Some(loc.def.krate), loc.def.local_inner)
                        }
                        MacroDefKind::BuiltIn(..) => (info, Some(loc.def.krate), false),
                        MacroDefKind::BuiltInDerive(..) => (info, None, false),
                        MacroDefKind::BuiltInEager(..) => (info, None, false),
                        MacroDefKind::ProcMacro(..) => (info, None, false),
                    }
                }
            },
        };

        let info = match info {
            None => {
                return HygieneFrame {
                    expansion: None,
                    local_inner,
                    krate,
                    call_site: None,
                    def_site: None,
                };
            }
            Some(it) => it,
        };

        let def_site = info.def_start.map(|it| db.hygiene_frame(it.file_id));
        let call_site = Some(db.hygiene_frame(info.arg_start.file_id));

        HygieneFrame { expansion: Some(info), local_inner, krate, call_site, def_site }
    }
}
