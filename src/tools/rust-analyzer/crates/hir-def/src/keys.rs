//! keys to be used with `DynMap`

use std::marker::PhantomData;

use hir_expand::{attrs::AttrId, MacroCallId};
use rustc_hash::FxHashMap;
use syntax::{ast, AstNode, AstPtr};

use crate::{
    dyn_map::{DynMap, Policy},
    ConstId, EnumId, EnumVariantId, FieldId, FunctionId, ImplId, LifetimeParamId, Macro2Id,
    MacroRulesId, ProcMacroId, StaticId, StructId, TraitId, TypeAliasId, TypeOrConstParamId,
    UnionId,
};

pub type Key<K, V> = crate::dyn_map::Key<K, V, AstPtrPolicy<K, V>>;

pub const FUNCTION: Key<ast::Fn, FunctionId> = Key::new();
pub const CONST: Key<ast::Const, ConstId> = Key::new();
pub const STATIC: Key<ast::Static, StaticId> = Key::new();
pub const TYPE_ALIAS: Key<ast::TypeAlias, TypeAliasId> = Key::new();
pub const IMPL: Key<ast::Impl, ImplId> = Key::new();
pub const TRAIT: Key<ast::Trait, TraitId> = Key::new();
pub const STRUCT: Key<ast::Struct, StructId> = Key::new();
pub const UNION: Key<ast::Union, UnionId> = Key::new();
pub const ENUM: Key<ast::Enum, EnumId> = Key::new();

pub const VARIANT: Key<ast::Variant, EnumVariantId> = Key::new();
pub const TUPLE_FIELD: Key<ast::TupleField, FieldId> = Key::new();
pub const RECORD_FIELD: Key<ast::RecordField, FieldId> = Key::new();
pub const TYPE_PARAM: Key<ast::TypeParam, TypeOrConstParamId> = Key::new();
pub const CONST_PARAM: Key<ast::ConstParam, TypeOrConstParamId> = Key::new();
pub const LIFETIME_PARAM: Key<ast::LifetimeParam, LifetimeParamId> = Key::new();

pub const MACRO_RULES: Key<ast::MacroRules, MacroRulesId> = Key::new();
pub const MACRO2: Key<ast::MacroDef, Macro2Id> = Key::new();
pub const PROC_MACRO: Key<ast::Fn, ProcMacroId> = Key::new();
pub const ATTR_MACRO_CALL: Key<ast::Item, MacroCallId> = Key::new();
pub const DERIVE_MACRO_CALL: Key<ast::Attr, (AttrId, MacroCallId, Box<[Option<MacroCallId>]>)> =
    Key::new();

/// XXX: AST Nodes and SyntaxNodes have identity equality semantics: nodes are
/// equal if they point to exactly the same object.
///
/// In general, we do not guarantee that we have exactly one instance of a
/// syntax tree for each file. We probably should add such guarantee, but, for
/// the time being, we will use identity-less AstPtr comparison.
pub struct AstPtrPolicy<AST, ID> {
    _phantom: PhantomData<(AST, ID)>,
}

impl<AST: AstNode + 'static, ID: 'static> Policy for AstPtrPolicy<AST, ID> {
    type K = AST;
    type V = ID;
    fn insert(map: &mut DynMap, key: AST, value: ID) {
        let key = AstPtr::new(&key);
        map.map
            .entry::<FxHashMap<AstPtr<AST>, ID>>()
            .or_insert_with(Default::default)
            .insert(key, value);
    }
    fn get<'a>(map: &'a DynMap, key: &AST) -> Option<&'a ID> {
        let key = AstPtr::new(key);
        map.map.get::<FxHashMap<AstPtr<AST>, ID>>()?.get(&key)
    }
    fn is_empty(map: &DynMap) -> bool {
        map.map.get::<FxHashMap<AstPtr<AST>, ID>>().map_or(true, |it| it.is_empty())
    }
}
