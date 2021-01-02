//! This modules handles hygiene information.
//!
//! Specifically, `ast` + `Hygiene` allows you to create a `Name`. Note that, at
//! this moment, this is horribly incomplete and handles only `$crate`.
use std::sync::Arc;

use arena::{Arena, Idx};
use base_db::CrateId;
use either::Either;
use mbe::Origin;
use syntax::{ast, AstNode};

use crate::{
    db::AstDatabase,
    name::{AsName, Name},
    ExpansionInfo, HirFileId, HirFileIdRepr, MacroCallId, MacroDefKind,
};

#[derive(Clone, Debug)]
pub struct Hygiene {
    frames: Option<Arc<HygieneFrames>>,
}

impl Hygiene {
    pub fn new(db: &dyn AstDatabase, file_id: HirFileId) -> Hygiene {
        Hygiene { frames: Some(Arc::new(HygieneFrames::new(db, file_id.clone()))) }
    }

    pub fn new_unhygienic() -> Hygiene {
        Hygiene { frames: None }
    }

    // FIXME: this should just return name
    pub fn name_ref_to_name(&self, name_ref: ast::NameRef) -> Either<Name, CrateId> {
        if let Some(frames) = &self.frames {
            if name_ref.text() == "$crate" {
                if let Some(krate) = frames.root_crate(&name_ref) {
                    return Either::Right(krate);
                }
            }
        }

        Either::Left(name_ref.as_name())
    }

    pub fn local_inner_macros(&self, path: ast::Path) -> Option<CrateId> {
        let frames = self.frames.as_ref()?;

        let mut token = path.syntax().first_token()?;
        let mut current = frames.first();

        while let Some((frame, data)) =
            current.and_then(|it| Some((it, it.expansion.as_ref()?.map_token_up(&token)?)))
        {
            let (mapped, origin) = data;
            if origin == Origin::Def {
                return if frame.local_inner { frame.krate } else { None };
            }
            current = Some(&frames.0[frame.call_site?]);
            token = mapped.value;
        }
        None
    }
}

#[derive(Default, Debug)]
struct HygieneFrames(Arena<HygieneFrame>);

#[derive(Clone, Debug)]
struct HygieneFrame {
    expansion: Option<ExpansionInfo>,

    // Indicate this is a local inner macro
    local_inner: bool,
    krate: Option<CrateId>,

    call_site: Option<Idx<HygieneFrame>>,
    def_site: Option<Idx<HygieneFrame>>,
}

impl HygieneFrames {
    fn new(db: &dyn AstDatabase, file_id: HirFileId) -> Self {
        let mut frames = HygieneFrames::default();
        frames.add(db, file_id);
        frames
    }

    fn add(&mut self, db: &dyn AstDatabase, file_id: HirFileId) -> Option<Idx<HygieneFrame>> {
        let (krate, local_inner) = match file_id.0 {
            HirFileIdRepr::FileId(_) => (None, false),
            HirFileIdRepr::MacroFile(macro_file) => match macro_file.macro_call_id {
                MacroCallId::EagerMacro(_id) => (None, false),
                MacroCallId::LazyMacro(id) => {
                    let loc = db.lookup_intern_macro(id);
                    match loc.def.kind {
                        MacroDefKind::Declarative => (Some(loc.def.krate), loc.def.local_inner),
                        MacroDefKind::BuiltIn(_) => (Some(loc.def.krate), false),
                        MacroDefKind::BuiltInDerive(_) => (None, false),
                        MacroDefKind::BuiltInEager(_) => (None, false),
                        MacroDefKind::ProcMacro(_) => (None, false),
                    }
                }
            },
        };

        let expansion = file_id.expansion_info(db);
        let expansion = match expansion {
            None => {
                return Some(self.0.alloc(HygieneFrame {
                    expansion: None,
                    local_inner,
                    krate,
                    call_site: None,
                    def_site: None,
                }));
            }
            Some(it) => it,
        };

        let def_site = expansion.def.clone();
        let call_site = expansion.arg.file_id;
        let idx = self.0.alloc(HygieneFrame {
            expansion: Some(expansion),
            local_inner,
            krate,
            call_site: None,
            def_site: None,
        });

        self.0[idx].call_site = self.add(db, call_site);
        self.0[idx].def_site = def_site.and_then(|it| self.add(db, it.file_id));

        Some(idx)
    }

    fn first(&self) -> Option<&HygieneFrame> {
        self.0.iter().next().map(|it| it.1)
    }

    fn root_crate(&self, name_ref: &ast::NameRef) -> Option<CrateId> {
        let mut token = name_ref.syntax().first_token()?;
        let first = self.first()?;
        let mut result = first.krate;
        let mut current = Some(first);

        while let Some((frame, (mapped, origin))) =
            current.and_then(|it| Some((it, it.expansion.as_ref()?.map_token_up(&token)?)))
        {
            result = frame.krate;

            let site = match origin {
                Origin::Def => frame.def_site,
                Origin::Call => frame.call_site,
            };

            let site = match site {
                None => break,
                Some(it) => it,
            };

            current = Some(&self.0[site]);
            token = mapped.value;
        }

        result
    }
}
