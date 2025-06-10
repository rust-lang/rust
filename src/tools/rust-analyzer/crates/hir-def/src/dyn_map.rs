//! This module defines a `DynMap` -- a container for heterogeneous maps.
//!
//! This means that `DynMap` stores a bunch of hash maps inside, and those maps
//! can be of different types.
//!
//! It is used like this:
//!
//! ```ignore
//! # use hir_def::dyn_map::DynMap;
//! # use hir_def::dyn_map::Key;
//! // keys define submaps of a `DynMap`
//! const STRING_TO_U32: Key<String, u32> = Key::new();
//! const U32_TO_VEC: Key<u32, Vec<bool>> = Key::new();
//!
//! // Note: concrete type, no type params!
//! let mut map = DynMap::new();
//!
//! // To access a specific map, index the `DynMap` by `Key`:
//! map[STRING_TO_U32].insert("hello".to_string(), 92);
//! let value = map[U32_TO_VEC].get(92);
//! assert!(value.is_none());
//! ```
//!
//! This is a work of fiction. Any similarities to Kotlin's `BindingContext` are
//! a coincidence.

pub mod keys {
    use std::marker::PhantomData;

    use hir_expand::{MacroCallId, attrs::AttrId};
    use rustc_hash::FxHashMap;
    use syntax::{AstNode, AstPtr, ast};

    use crate::{
        BlockId, ConstId, EnumId, EnumVariantId, ExternBlockId, ExternCrateId, FieldId, FunctionId,
        ImplId, LifetimeParamId, Macro2Id, MacroRulesId, ProcMacroId, StaticId, StructId,
        TraitAliasId, TraitId, TypeAliasId, TypeOrConstParamId, UnionId, UseId,
        dyn_map::{DynMap, Policy},
    };

    pub type Key<K, V> = crate::dyn_map::Key<AstPtr<K>, V, AstPtrPolicy<K, V>>;

    pub const BLOCK: Key<ast::BlockExpr, BlockId> = Key::new();
    pub const FUNCTION: Key<ast::Fn, FunctionId> = Key::new();
    pub const CONST: Key<ast::Const, ConstId> = Key::new();
    pub const STATIC: Key<ast::Static, StaticId> = Key::new();
    pub const TYPE_ALIAS: Key<ast::TypeAlias, TypeAliasId> = Key::new();
    pub const IMPL: Key<ast::Impl, ImplId> = Key::new();
    pub const EXTERN_BLOCK: Key<ast::ExternBlock, ExternBlockId> = Key::new();
    pub const TRAIT: Key<ast::Trait, TraitId> = Key::new();
    pub const TRAIT_ALIAS: Key<ast::TraitAlias, TraitAliasId> = Key::new();
    pub const STRUCT: Key<ast::Struct, StructId> = Key::new();
    pub const UNION: Key<ast::Union, UnionId> = Key::new();
    pub const ENUM: Key<ast::Enum, EnumId> = Key::new();
    pub const EXTERN_CRATE: Key<ast::ExternCrate, ExternCrateId> = Key::new();
    pub const USE: Key<ast::Use, UseId> = Key::new();

    pub const ENUM_VARIANT: Key<ast::Variant, EnumVariantId> = Key::new();
    pub const TUPLE_FIELD: Key<ast::TupleField, FieldId> = Key::new();
    pub const RECORD_FIELD: Key<ast::RecordField, FieldId> = Key::new();
    pub const TYPE_PARAM: Key<ast::TypeParam, TypeOrConstParamId> = Key::new();
    pub const CONST_PARAM: Key<ast::ConstParam, TypeOrConstParamId> = Key::new();
    pub const LIFETIME_PARAM: Key<ast::LifetimeParam, LifetimeParamId> = Key::new();

    pub const MACRO_RULES: Key<ast::MacroRules, MacroRulesId> = Key::new();
    pub const MACRO2: Key<ast::MacroDef, Macro2Id> = Key::new();
    pub const PROC_MACRO: Key<ast::Fn, ProcMacroId> = Key::new();
    pub const MACRO_CALL: Key<ast::MacroCall, MacroCallId> = Key::new();
    pub const ATTR_MACRO_CALL: Key<ast::Item, MacroCallId> = Key::new();
    pub const DERIVE_MACRO_CALL: Key<
        ast::Attr,
        (
            AttrId,
            /* derive() */ MacroCallId,
            /* actual derive macros */ Box<[Option<MacroCallId>]>,
        ),
    > = Key::new();

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
        type K = AstPtr<AST>;
        type V = ID;
        fn insert(map: &mut DynMap, key: AstPtr<AST>, value: ID) {
            map.map
                .entry::<FxHashMap<AstPtr<AST>, ID>>()
                .or_insert_with(Default::default)
                .insert(key, value);
        }
        fn get<'a>(map: &'a DynMap, key: &AstPtr<AST>) -> Option<&'a ID> {
            map.map.get::<FxHashMap<AstPtr<AST>, ID>>()?.get(key)
        }
        fn is_empty(map: &DynMap) -> bool {
            map.map.get::<FxHashMap<AstPtr<AST>, ID>>().is_none_or(|it| it.is_empty())
        }
    }
}

use std::{
    hash::Hash,
    marker::PhantomData,
    ops::{Index, IndexMut},
};

use rustc_hash::FxHashMap;
use stdx::anymap::Map;

pub struct Key<K, V, P = (K, V)> {
    _phantom: PhantomData<(K, V, P)>,
}

impl<K, V, P> Key<K, V, P> {
    #[allow(
        clippy::new_without_default,
        reason = "this a const fn, so it can't be default yet. See <https://github.com/rust-lang/rust/issues/63065>"
    )]
    pub(crate) const fn new() -> Key<K, V, P> {
        Key { _phantom: PhantomData }
    }
}

impl<K, V, P> Copy for Key<K, V, P> {}

impl<K, V, P> Clone for Key<K, V, P> {
    fn clone(&self) -> Key<K, V, P> {
        *self
    }
}

pub trait Policy {
    type K;
    type V;

    fn insert(map: &mut DynMap, key: Self::K, value: Self::V);
    fn get<'a>(map: &'a DynMap, key: &Self::K) -> Option<&'a Self::V>;
    fn is_empty(map: &DynMap) -> bool;
}

impl<K: Hash + Eq + 'static, V: 'static> Policy for (K, V) {
    type K = K;
    type V = V;
    fn insert(map: &mut DynMap, key: K, value: V) {
        map.map.entry::<FxHashMap<K, V>>().or_insert_with(Default::default).insert(key, value);
    }
    fn get<'a>(map: &'a DynMap, key: &K) -> Option<&'a V> {
        map.map.get::<FxHashMap<K, V>>()?.get(key)
    }
    fn is_empty(map: &DynMap) -> bool {
        map.map.get::<FxHashMap<K, V>>().is_none_or(|it| it.is_empty())
    }
}

#[derive(Default)]
pub struct DynMap {
    pub(crate) map: Map,
}

#[repr(transparent)]
pub struct KeyMap<KEY> {
    map: DynMap,
    _phantom: PhantomData<KEY>,
}

impl<P: Policy> KeyMap<Key<P::K, P::V, P>> {
    pub fn insert(&mut self, key: P::K, value: P::V) {
        P::insert(&mut self.map, key, value)
    }
    pub fn get(&self, key: &P::K) -> Option<&P::V> {
        P::get(&self.map, key)
    }

    pub fn is_empty(&self) -> bool {
        P::is_empty(&self.map)
    }
}

impl<P: Policy> Index<Key<P::K, P::V, P>> for DynMap {
    type Output = KeyMap<Key<P::K, P::V, P>>;
    fn index(&self, _key: Key<P::K, P::V, P>) -> &Self::Output {
        // Safe due to `#[repr(transparent)]`.
        unsafe { std::mem::transmute::<&DynMap, &KeyMap<Key<P::K, P::V, P>>>(self) }
    }
}

impl<P: Policy> IndexMut<Key<P::K, P::V, P>> for DynMap {
    fn index_mut(&mut self, _key: Key<P::K, P::V, P>) -> &mut Self::Output {
        // Safe due to `#[repr(transparent)]`.
        unsafe { std::mem::transmute::<&mut DynMap, &mut KeyMap<Key<P::K, P::V, P>>>(self) }
    }
}
