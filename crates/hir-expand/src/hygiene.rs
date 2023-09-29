//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use base_db::CrateId;
use db::TokenExpander;
use either::Either;
use syntax::{
    ast::{self, HasDocComments},
    AstNode, SyntaxNode, TextRange, TextSize,
};
use triomphe::Arc;

use crate::{
    db::{self, ExpandDatabase},
    name::{AsName, Name},
    HirFileId, InFile, MacroCallKind, MacroCallLoc, MacroDefKind, MacroFile, SpanMap,
};

#[derive(Clone, Debug)]
pub struct Hygiene {
    frames: Option<HygieneFrames>,
}

impl Hygiene {
    pub fn new(db: &dyn ExpandDatabase, file_id: HirFileId) -> Hygiene {
        Hygiene { frames: Some(HygieneFrames::new(db, file_id)) }
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene { frames: None }
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(
        &self,
        db: &dyn ExpandDatabase,
        name_ref: ast::NameRef,
    ) -> Either<Name, CrateId> {
        if let Some(frames) = &self.frames {
            if name_ref.text() == "$crate" {
                if let Some(krate) = frames.root_crate(db, name_ref.syntax()) {
                    return Either::Right(krate);
                }
            }
        }

        Either::Left(name_ref.as_name())
    }

    pub fn local_inner_macros(&self, _db: &dyn ExpandDatabase, path: ast::Path) -> Option<CrateId> {
        let mut _token = path.syntax().first_token()?.text_range();
        let frames = self.frames.as_ref()?;
        let mut _current = &frames.0;

        // FIXME: Hygiene ...
        return None;
        // loop {
        //     let (mapped, origin) = current.expansion.as_ref()?.map_ident_up(db, token)?;
        //     if origin == Origin::Def {
        //         return if current.local_inner {
        //             frames.root_crate(db, path.syntax())
        //         } else {
        //             None
        //         };
        //     }
        //     current = current.call_site.as_ref()?;
        //     token = mapped.value;
        // }
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
    fn new(db: &dyn ExpandDatabase, file_id: HirFileId) -> Self {
        // Note that this intentionally avoids the `hygiene_frame` query to avoid blowing up memory
        // usage. The query is only helpful for nested `HygieneFrame`s as it avoids redundant work.
        HygieneFrames(Arc::new(HygieneFrame::new(db, file_id)))
    }

    fn root_crate(&self, _db: &dyn ExpandDatabase, node: &SyntaxNode) -> Option<CrateId> {
        let mut _token = node.first_token()?.text_range();
        let mut _result = self.0.krate;
        let mut _current = self.0.clone();

        return None;

        //     while let Some((mapped, origin)) =
        //         current.expansion.as_ref().and_then(|it| it.map_ident_up(db, token))
        //     {
        //         result = current.krate;

        //         let site = match origin {
        //             Origin::Def => &current.def_site,
        //             Origin::Call => &current.call_site,
        //         };

        //         let site = match site {
        //             None => break,
        //             Some(it) => it,
        //         };

        //         current = site.clone();
        //         token = mapped.value;
        //     }

        //     result
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HygieneInfo {
    file: MacroFile,
    /// The start offset of the `macro_rules!` arguments or attribute input.
    attr_input_or_mac_def_start: Option<InFile<TextSize>>,

    macro_def: TokenExpander,
    macro_arg: Arc<crate::tt::Subtree>,
    exp_map: Arc<SpanMap>,
}

impl HygieneInfo {
    fn _map_ident_up(
        &self,
        _db: &dyn ExpandDatabase,
        _token: TextRange,
    ) -> Option<InFile<TextRange>> {
        // self.exp_map.token_by_range(token).map(|span| InFile::new(span.anchor, span.range))
        None
    }
}

fn make_hygiene_info(
    db: &dyn ExpandDatabase,
    macro_file: MacroFile,
    loc: &MacroCallLoc,
) -> HygieneInfo {
    let def = loc.def.ast_id().left().and_then(|id| {
        let def_tt = match id.to_node(db) {
            ast::Macro::MacroRules(mac) => mac.token_tree()?,
            ast::Macro::MacroDef(mac) => mac.body()?,
        };
        Some(InFile::new(id.file_id, def_tt))
    });
    let attr_input_or_mac_def = def.or_else(|| match loc.kind {
        MacroCallKind::Attr { ast_id, invoc_attr_index, .. } => {
            let tt = ast_id
                .to_node(db)
                .doc_comments_and_attrs()
                .nth(invoc_attr_index.ast_index())
                .and_then(Either::left)?
                .token_tree()?;
            Some(InFile::new(ast_id.file_id, tt))
        }
        _ => None,
    });

    let macro_def = db.macro_expander(loc.def);
    let (_, exp_map) = db.parse_macro_expansion(macro_file).value;
    let macro_arg = db.macro_arg(macro_file.macro_call_id).value.unwrap_or_else(|| {
        Arc::new(tt::Subtree { delimiter: tt::Delimiter::UNSPECIFIED, token_trees: Vec::new() })
    });

    HygieneInfo {
        file: macro_file,
        attr_input_or_mac_def_start: attr_input_or_mac_def
            .map(|it| it.map(|tt| tt.syntax().text_range().start())),
        macro_arg,
        macro_def,
        exp_map,
    }
}

impl HygieneFrame {
    pub(crate) fn new(db: &dyn ExpandDatabase, file_id: HirFileId) -> HygieneFrame {
        let (info, krate, local_inner) = match file_id.macro_file() {
            None => (None, None, false),
            Some(macro_file) => {
                let loc = db.lookup_intern_macro_call(macro_file.macro_call_id);
                let info = Some((make_hygiene_info(db, macro_file, &loc), loc.kind.file_id()));
                match loc.def.kind {
                    MacroDefKind::Declarative(_) => {
                        (info, Some(loc.def.krate), loc.def.local_inner)
                    }
                    MacroDefKind::BuiltIn(..) => (info, Some(loc.def.krate), false),
                    MacroDefKind::BuiltInAttr(..) => (info, None, false),
                    MacroDefKind::BuiltInDerive(..) => (info, None, false),
                    MacroDefKind::BuiltInEager(..) => (info, None, false),
                    MacroDefKind::ProcMacro(..) => (info, None, false),
                }
            }
        };

        let Some((info, calling_file)) = info else {
            return HygieneFrame {
                expansion: None,
                local_inner,
                krate,
                call_site: None,
                def_site: None,
            };
        };

        let def_site = info.attr_input_or_mac_def_start.map(|it| db.hygiene_frame(it.file_id));
        let call_site = Some(db.hygiene_frame(calling_file));

        HygieneFrame { expansion: Some(info), local_inner, krate, call_site, def_site }
    }
}
