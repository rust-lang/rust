//! FIXME: write short doc here

use hir_def::{AstItemDef, LocationCtx, ModuleId};
use hir_expand::{name::AsName, AstId, MacroDefId, MacroDefKind};
use ra_syntax::{
    ast::{self, AstNode, NameOwner},
    match_ast, AstPtr, SyntaxNode,
};

use crate::{
    db::{AstDatabase, DefDatabase, HirDatabase},
    AssocItem, Const, DefWithBody, Enum, EnumVariant, FieldSource, Function, HasSource, ImplBlock,
    InFile, Local, MacroDef, Module, ModuleDef, ModuleSource, Static, Struct, StructField, Trait,
    TypeAlias, Union, VariantDef,
};

pub trait FromSource: Sized {
    type Ast;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self>;
}

impl FromSource for Struct {
    type Ast = ast::StructDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Struct { id })
    }
}
impl FromSource for Union {
    type Ast = ast::UnionDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Union { id })
    }
}
impl FromSource for Enum {
    type Ast = ast::EnumDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Enum { id })
    }
}
impl FromSource for Trait {
    type Ast = ast::TraitDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(Trait { id })
    }
}
impl FromSource for Function {
    type Ast = ast::FnDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let items = match Container::find(db, src.as_ref().map(|it| it.syntax()))? {
            Container::Trait(it) => it.items(db),
            Container::ImplBlock(it) => it.items(db),
            Container::Module(m) => {
                return m
                    .declarations(db)
                    .into_iter()
                    .filter_map(|it| match it {
                        ModuleDef::Function(it) => Some(it),
                        _ => None,
                    })
                    .find(|it| same_source(&it.source(db), &src))
            }
        };
        items
            .into_iter()
            .filter_map(|it| match it {
                AssocItem::Function(it) => Some(it),
                _ => None,
            })
            .find(|it| same_source(&it.source(db), &src))
    }
}

impl FromSource for Const {
    type Ast = ast::ConstDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let items = match Container::find(db, src.as_ref().map(|it| it.syntax()))? {
            Container::Trait(it) => it.items(db),
            Container::ImplBlock(it) => it.items(db),
            Container::Module(m) => {
                return m
                    .declarations(db)
                    .into_iter()
                    .filter_map(|it| match it {
                        ModuleDef::Const(it) => Some(it),
                        _ => None,
                    })
                    .find(|it| same_source(&it.source(db), &src))
            }
        };
        items
            .into_iter()
            .filter_map(|it| match it {
                AssocItem::Const(it) => Some(it),
                _ => None,
            })
            .find(|it| same_source(&it.source(db), &src))
    }
}
impl FromSource for Static {
    type Ast = ast::StaticDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let module = match Container::find(db, src.as_ref().map(|it| it.syntax()))? {
            Container::Module(it) => it,
            Container::Trait(_) | Container::ImplBlock(_) => return None,
        };
        module
            .declarations(db)
            .into_iter()
            .filter_map(|it| match it {
                ModuleDef::Static(it) => Some(it),
                _ => None,
            })
            .find(|it| same_source(&it.source(db), &src))
    }
}

impl FromSource for TypeAlias {
    type Ast = ast::TypeAliasDef;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let items = match Container::find(db, src.as_ref().map(|it| it.syntax()))? {
            Container::Trait(it) => it.items(db),
            Container::ImplBlock(it) => it.items(db),
            Container::Module(m) => {
                return m
                    .declarations(db)
                    .into_iter()
                    .filter_map(|it| match it {
                        ModuleDef::TypeAlias(it) => Some(it),
                        _ => None,
                    })
                    .find(|it| same_source(&it.source(db), &src))
            }
        };
        items
            .into_iter()
            .filter_map(|it| match it {
                AssocItem::TypeAlias(it) => Some(it),
                _ => None,
            })
            .find(|it| same_source(&it.source(db), &src))
    }
}

impl FromSource for MacroDef {
    type Ast = ast::MacroCall;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let kind = MacroDefKind::Declarative;

        let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
        let module = Module::from_definition(db, InFile::new(src.file_id, module_src))?;
        let krate = module.krate().crate_id();

        let ast_id = AstId::new(src.file_id, db.ast_id_map(src.file_id).ast_id(&src.value));

        let id: MacroDefId = MacroDefId { krate, ast_id, kind };
        Some(MacroDef { id })
    }
}

impl FromSource for ImplBlock {
    type Ast = ast::ImplBlock;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let id = from_source(db, src)?;
        Some(ImplBlock { id })
    }
}

impl FromSource for EnumVariant {
    type Ast = ast::EnumVariant;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let parent_enum = src.value.parent_enum();
        let src_enum = InFile { file_id: src.file_id, value: parent_enum };
        let variants = Enum::from_source(db, src_enum)?.variants(db);
        variants.into_iter().find(|v| same_source(&v.source(db), &src))
    }
}

impl FromSource for StructField {
    type Ast = FieldSource;
    fn from_source(db: &(impl DefDatabase + AstDatabase), src: InFile<Self::Ast>) -> Option<Self> {
        let variant_def: VariantDef = match src.value {
            FieldSource::Named(ref field) => {
                let value = field.syntax().ancestors().find_map(ast::StructDef::cast)?;
                let src = InFile { file_id: src.file_id, value };
                let def = Struct::from_source(db, src)?;
                VariantDef::from(def)
            }
            FieldSource::Pos(ref field) => {
                let value = field.syntax().ancestors().find_map(ast::EnumVariant::cast)?;
                let src = InFile { file_id: src.file_id, value };
                let def = EnumVariant::from_source(db, src)?;
                VariantDef::from(def)
            }
        };
        variant_def
            .variant_data(db)
            .fields()
            .iter()
            .map(|(id, _)| StructField { parent: variant_def, id })
            .find(|f| f.source(db) == src)
    }
}

impl Local {
    pub fn from_source(db: &impl HirDatabase, src: InFile<ast::BindPat>) -> Option<Self> {
        let file_id = src.file_id;
        let parent: DefWithBody = src.value.syntax().ancestors().find_map(|it| {
            let res = match_ast! {
                match it {
                    ast::ConstDef(value) => { Const::from_source(db, InFile { value, file_id})?.into() },
                    ast::StaticDef(value) => { Static::from_source(db, InFile { value, file_id})?.into() },
                    ast::FnDef(value) => { Function::from_source(db, InFile { value, file_id})?.into() },
                    _ => return None,
                }
            };
            Some(res)
        })?;
        let (_body, source_map) = db.body_with_source_map(parent.into());
        let src = src.map(ast::Pat::from);
        let pat_id = source_map.node_pat(src.as_ref())?;
        Some(Local { parent, pat_id })
    }
}

impl Module {
    pub fn from_declaration(db: &impl DefDatabase, src: InFile<ast::Module>) -> Option<Self> {
        let parent_declaration = src.value.syntax().ancestors().skip(1).find_map(ast::Module::cast);

        let parent_module = match parent_declaration {
            Some(parent_declaration) => {
                let src_parent = InFile { file_id: src.file_id, value: parent_declaration };
                Module::from_declaration(db, src_parent)
            }
            _ => {
                let src_parent = InFile {
                    file_id: src.file_id,
                    value: ModuleSource::new(db, Some(src.file_id.original_file(db)), None),
                };
                Module::from_definition(db, src_parent)
            }
        }?;

        let child_name = src.value.name()?;
        parent_module.child(db, &child_name.as_name())
    }

    pub fn from_definition(db: &impl DefDatabase, src: InFile<ModuleSource>) -> Option<Self> {
        match src.value {
            ModuleSource::Module(ref module) => {
                assert!(!module.has_semi());
                return Module::from_declaration(
                    db,
                    InFile { file_id: src.file_id, value: module.clone() },
                );
            }
            ModuleSource::SourceFile(_) => (),
        };

        let original_file = src.file_id.original_file(db);

        let (krate, local_id) = db.relevant_crates(original_file).iter().find_map(|&crate_id| {
            let crate_def_map = db.crate_def_map(crate_id);
            let local_id = crate_def_map.modules_for_file(original_file).next()?;
            Some((crate_id, local_id))
        })?;
        Some(Module { id: ModuleId { krate, local_id } })
    }
}

fn from_source<N, DEF>(db: &(impl DefDatabase + AstDatabase), src: InFile<N>) -> Option<DEF>
where
    N: AstNode,
    DEF: AstItemDef<N>,
{
    let module_src = ModuleSource::from_child_node(db, src.as_ref().map(|it| it.syntax()));
    let module = Module::from_definition(db, InFile::new(src.file_id, module_src))?;
    let ctx = LocationCtx::new(db, module.id, src.file_id);
    let items = db.ast_id_map(src.file_id);
    let item_id = items.ast_id(&src.value);
    Some(DEF::from_ast_id(ctx, item_id))
}

enum Container {
    Trait(Trait),
    ImplBlock(ImplBlock),
    Module(Module),
}

impl Container {
    fn find(db: &impl DefDatabase, src: InFile<&SyntaxNode>) -> Option<Container> {
        // FIXME: this doesn't try to handle nested declarations
        for container in src.value.ancestors() {
            let res = match_ast! {
                match container {
                    ast::TraitDef(it) => {
                        let c = Trait::from_source(db, src.with_value(it))?;
                        Container::Trait(c)
                    },
                    ast::ImplBlock(it) => {
                        let c = ImplBlock::from_source(db, src.with_value(it))?;
                        Container::ImplBlock(c)
                     },
                    _ => { continue },
                }
            };
            return Some(res);
        }

        let module_source = ModuleSource::from_child_node(db, src);
        let c = Module::from_definition(db, src.with_value(module_source))?;
        Some(Container::Module(c))
    }
}

/// XXX: AST Nodes and SyntaxNodes have identity equality semantics: nodes are
/// equal if they point to exactly the same object.
///
/// In general, we do not guarantee that we have exactly one instance of a
/// syntax tree for each file. We probably should add such guarantee, but, for
/// the time being, we will use identity-less AstPtr comparison.
fn same_source<N: AstNode>(s1: &InFile<N>, s2: &InFile<N>) -> bool {
    s1.as_ref().map(AstPtr::new) == s2.as_ref().map(AstPtr::new)
}
