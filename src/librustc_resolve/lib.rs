// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "rustc_resolve"]
#![unstable(feature = "rustc_private")]
#![staged_api]
#![crate_type = "dylib"]
#![crate_type = "rlib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
      html_favicon_url = "http://www.rust-lang.org/favicon.ico",
      html_root_url = "http://doc.rust-lang.org/nightly/")]

#![feature(alloc)]
#![feature(collections)]
#![feature(core)]
#![feature(int_uint)]
#![feature(rustc_diagnostic_macros)]
#![feature(rustc_private)]
#![feature(staged_api)]
#![feature(std_misc)]

#[macro_use] extern crate log;
#[macro_use] extern crate syntax;
#[macro_use] #[no_link] extern crate rustc_bitflags;

extern crate rustc;

use self::PatternBindingMode::*;
use self::Namespace::*;
use self::NamespaceResult::*;
use self::NameDefinition::*;
use self::ImportDirectiveSubclass::*;
use self::ResolveResult::*;
use self::FallbackSuggestion::*;
use self::TypeParameters::*;
use self::RibKind::*;
use self::MethodSort::*;
use self::UseLexicalScopeFlag::*;
use self::ModulePrefixResult::*;
use self::NameSearchType::*;
use self::BareIdentifierPatternResolution::*;
use self::ParentLink::*;
use self::ModuleKind::*;
use self::TraitReferenceType::*;
use self::FallbackChecks::*;

use rustc::session::Session;
use rustc::lint;
use rustc::metadata::csearch;
use rustc::metadata::decoder::{DefLike, DlDef, DlField, DlImpl};
use rustc::middle::def::*;
use rustc::middle::lang_items::LanguageItems;
use rustc::middle::pat_util::pat_bindings;
use rustc::middle::privacy::*;
use rustc::middle::subst::{ParamSpace, FnSpace, TypeSpace};
use rustc::middle::ty::{Freevar, FreevarMap, TraitMap, GlobMap};
use rustc::util::nodemap::{NodeMap, NodeSet, DefIdSet, FnvHashMap};
use rustc::util::lev_distance::lev_distance;

use syntax::ast::{Arm, BindByRef, BindByValue, BindingMode, Block, Crate, CrateNum};
use syntax::ast::{DefId, Expr, ExprAgain, ExprBreak, ExprField};
use syntax::ast::{ExprClosure, ExprLoop, ExprWhile, ExprMethodCall};
use syntax::ast::{ExprPath, ExprQPath, ExprStruct, FnDecl};
use syntax::ast::{ForeignItemFn, ForeignItemStatic, Generics};
use syntax::ast::{Ident, ImplItem, Item, ItemConst, ItemEnum, ItemExternCrate};
use syntax::ast::{ItemFn, ItemForeignMod, ItemImpl, ItemMac, ItemMod, ItemStatic};
use syntax::ast::{ItemStruct, ItemTrait, ItemTy, ItemUse};
use syntax::ast::{Local, MethodImplItem, Mod, Name, NodeId};
use syntax::ast::{Pat, PatEnum, PatIdent, PatLit};
use syntax::ast::{PatRange, PatStruct, Path};
use syntax::ast::{PolyTraitRef, PrimTy, SelfExplicit};
use syntax::ast::{RegionTyParamBound, StructField};
use syntax::ast::{TraitRef, TraitTyParamBound};
use syntax::ast::{Ty, TyBool, TyChar, TyF32};
use syntax::ast::{TyF64, TyFloat, TyIs, TyI8, TyI16, TyI32, TyI64, TyInt, TyObjectSum};
use syntax::ast::{TyParam, TyParamBound, TyPath, TyPtr, TyPolyTraitRef, TyQPath};
use syntax::ast::{TyRptr, TyStr, TyUs, TyU8, TyU16, TyU32, TyU64, TyUint};
use syntax::ast::{TypeImplItem};
use syntax::ast;
use syntax::ast_map;
use syntax::ast_util::{PostExpansionMethod, local_def, walk_pat};
use syntax::attr::AttrMetaMethods;
use syntax::ext::mtwt;
use syntax::parse::token::{self, special_names, special_idents};
use syntax::codemap::{Span, Pos};
use syntax::owned_slice::OwnedSlice;
use syntax::visit::{self, Visitor};

use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::cell::{Cell, RefCell};
use std::fmt;
use std::mem::replace;
use std::rc::{Rc, Weak};
use std::usize;

// NB: This module needs to be declared first so diagnostics are
// registered before they are used.
pub mod diagnostics;

mod check_unused;
mod record_exports;
mod build_reduced_graph;

#[derive(Copy)]
struct BindingInfo {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<Name, BindingInfo>;

#[derive(Copy, PartialEq)]
enum PatternBindingMode {
    RefutableMode,
    LocalIrrefutableMode,
    ArgumentIrrefutableMode,
}

#[derive(Copy, PartialEq, Eq, Hash, Debug)]
enum Namespace {
    TypeNS,
    ValueNS
}

/// A NamespaceResult represents the result of resolving an import in
/// a particular namespace. The result is either definitely-resolved,
/// definitely- unresolved, or unknown.
#[derive(Clone)]
enum NamespaceResult {
    /// Means that resolve hasn't gathered enough information yet to determine
    /// whether the name is bound in this namespace. (That is, it hasn't
    /// resolved all `use` directives yet.)
    UnknownResult,
    /// Means that resolve has determined that the name is definitely
    /// not bound in the namespace.
    UnboundResult,
    /// Means that resolve has determined that the name is bound in the Module
    /// argument, and specified by the NameBindings argument.
    BoundResult(Rc<Module>, Rc<NameBindings>)
}

impl NamespaceResult {
    fn is_unknown(&self) -> bool {
        match *self {
            UnknownResult => true,
            _ => false
        }
    }
    fn is_unbound(&self) -> bool {
        match *self {
            UnboundResult => true,
            _ => false
        }
    }
}

enum NameDefinition {
    NoNameDefinition,           //< The name was unbound.
    ChildNameDefinition(Def, LastPrivate), //< The name identifies an immediate child.
    ImportNameDefinition(Def, LastPrivate) //< The name identifies an import.
}

impl<'a, 'v, 'tcx> Visitor<'v> for Resolver<'a, 'tcx> {
    fn visit_item(&mut self, item: &Item) {
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm: &Arm) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &Block) {
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr: &Expr) {
        self.resolve_expr(expr);
    }
    fn visit_local(&mut self, local: &Local) {
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty: &Ty) {
        self.resolve_type(ty);
    }
}

/// Contains data for specific types of import directives.
#[derive(Copy,Debug)]
enum ImportDirectiveSubclass {
    SingleImport(Name /* target */, Name /* source */),
    GlobImport
}

type ErrorMessage = Option<(Span, String)>;

enum ResolveResult<T> {
    Failed(ErrorMessage),   // Failed to resolve the name, optional helpful error message.
    Indeterminate,          // Couldn't determine due to unresolved globs.
    Success(T)              // Successfully resolved the import.
}

impl<T> ResolveResult<T> {
    fn indeterminate(&self) -> bool {
        match *self { Indeterminate => true, _ => false }
    }
}

enum FallbackSuggestion {
    NoSuggestion,
    Field,
    Method,
    TraitItem,
    StaticMethod(String),
    TraitMethod(String),
}

#[derive(Copy)]
enum TypeParameters<'a> {
    NoTypeParameters,
    HasTypeParameters(
        // Type parameters.
        &'a Generics,

        // Identifies the things that these parameters
        // were declared on (type, fn, etc)
        ParamSpace,

        // ID of the enclosing item.
        NodeId,

        // The kind of the rib used for type parameters.
        RibKind)
}

// The rib kind controls the translation of local
// definitions (`DefLocal`) to upvars (`DefUpvar`).
#[derive(Copy, Debug)]
enum RibKind {
    // No translation needs to be applied.
    NormalRibKind,

    // We passed through a closure scope at the given node ID.
    // Translate upvars as appropriate.
    ClosureRibKind(NodeId /* func id */),

    // We passed through an impl or trait and are now in one of its
    // methods. Allow references to ty params that impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
              // parent;   method itself
    MethodRibKind(NodeId, MethodSort),

    // We passed through an item scope. Disallow upvars.
    ItemRibKind,

    // We're in a constant item. Can't refer to dynamic stuff.
    ConstantItemRibKind
}

// Methods can be required or provided. RequiredMethod methods only occur in traits.
#[derive(Copy, Debug)]
enum MethodSort {
    RequiredMethod,
    ProvidedMethod(NodeId)
}

#[derive(Copy)]
enum UseLexicalScopeFlag {
    DontUseLexicalScope,
    UseLexicalScope
}

enum ModulePrefixResult {
    NoPrefixFound,
    PrefixFound(Rc<Module>, uint)
}

#[derive(Copy, PartialEq)]
enum NameSearchType {
    /// We're doing a name search in order to resolve a `use` directive.
    ImportSearch,

    /// We're doing a name search in order to resolve a path type, a path
    /// expression, or a path pattern.
    PathSearch,
}

#[derive(Copy)]
enum BareIdentifierPatternResolution {
    FoundStructOrEnumVariant(Def, LastPrivate),
    FoundConst(Def, LastPrivate),
    BareIdentifierPatternUnresolved
}

/// One local scope.
#[derive(Debug)]
struct Rib {
    bindings: HashMap<Name, DefLike>,
    kind: RibKind,
}

impl Rib {
    fn new(kind: RibKind) -> Rib {
        Rib {
            bindings: HashMap::new(),
            kind: kind
        }
    }
}

/// Whether an import can be shadowed by another import.
#[derive(Debug,PartialEq,Clone,Copy)]
enum Shadowable {
    Always,
    Never
}

/// One import directive.
#[derive(Debug)]
struct ImportDirective {
    module_path: Vec<Name>,
    subclass: ImportDirectiveSubclass,
    span: Span,
    id: NodeId,
    is_public: bool, // see note in ImportResolution about how to use this
    shadowable: Shadowable,
}

impl ImportDirective {
    fn new(module_path: Vec<Name> ,
           subclass: ImportDirectiveSubclass,
           span: Span,
           id: NodeId,
           is_public: bool,
           shadowable: Shadowable)
           -> ImportDirective {
        ImportDirective {
            module_path: module_path,
            subclass: subclass,
            span: span,
            id: id,
            is_public: is_public,
            shadowable: shadowable,
        }
    }
}

/// The item that an import resolves to.
#[derive(Clone,Debug)]
struct Target {
    target_module: Rc<Module>,
    bindings: Rc<NameBindings>,
    shadowable: Shadowable,
}

impl Target {
    fn new(target_module: Rc<Module>,
           bindings: Rc<NameBindings>,
           shadowable: Shadowable)
           -> Target {
        Target {
            target_module: target_module,
            bindings: bindings,
            shadowable: shadowable,
        }
    }
}

/// An ImportResolution represents a particular `use` directive.
#[derive(Debug)]
struct ImportResolution {
    /// Whether this resolution came from a `use` or a `pub use`. Note that this
    /// should *not* be used whenever resolution is being performed, this is
    /// only looked at for glob imports statements currently. Privacy testing
    /// occurs during a later phase of compilation.
    is_public: bool,

    // The number of outstanding references to this name. When this reaches
    // zero, outside modules can count on the targets being correct. Before
    // then, all bets are off; future imports could override this name.
    outstanding_references: uint,

    /// The value that this `use` directive names, if there is one.
    value_target: Option<Target>,
    /// The source node of the `use` directive leading to the value target
    /// being non-none
    value_id: NodeId,

    /// The type that this `use` directive names, if there is one.
    type_target: Option<Target>,
    /// The source node of the `use` directive leading to the type target
    /// being non-none
    type_id: NodeId,
}

impl ImportResolution {
    fn new(id: NodeId, is_public: bool) -> ImportResolution {
        ImportResolution {
            type_id: id,
            value_id: id,
            outstanding_references: 0,
            value_target: None,
            type_target: None,
            is_public: is_public,
        }
    }

    fn target_for_namespace(&self, namespace: Namespace)
                                -> Option<Target> {
        match namespace {
            TypeNS  => self.type_target.clone(),
            ValueNS => self.value_target.clone(),
        }
    }

    fn id(&self, namespace: Namespace) -> NodeId {
        match namespace {
            TypeNS  => self.type_id,
            ValueNS => self.value_id,
        }
    }

    fn shadowable(&self, namespace: Namespace) -> Shadowable {
        let target = self.target_for_namespace(namespace);
        if target.is_none() {
            return Shadowable::Always;
        }

        target.unwrap().shadowable
    }

    fn set_target_and_id(&mut self,
                         namespace: Namespace,
                         target: Option<Target>,
                         id: NodeId) {
        match namespace {
            TypeNS  => {
                self.type_target = target;
                self.type_id = id;
            }
            ValueNS => {
                self.value_target = target;
                self.value_id = id;
            }
        }
    }
}

/// The link from a module up to its nearest parent node.
#[derive(Clone,Debug)]
enum ParentLink {
    NoParentLink,
    ModuleParentLink(Weak<Module>, Name),
    BlockParentLink(Weak<Module>, NodeId)
}

/// The type of module this is.
#[derive(Copy, PartialEq, Debug)]
enum ModuleKind {
    NormalModuleKind,
    TraitModuleKind,
    ImplModuleKind,
    EnumModuleKind,
    TypeModuleKind,
    AnonymousModuleKind,
}

/// One node in the tree of modules.
struct Module {
    parent_link: ParentLink,
    def_id: Cell<Option<DefId>>,
    kind: Cell<ModuleKind>,
    is_public: bool,

    children: RefCell<HashMap<Name, Rc<NameBindings>>>,
    imports: RefCell<Vec<ImportDirective>>,

    // The external module children of this node that were declared with
    // `extern crate`.
    external_module_children: RefCell<HashMap<Name, Rc<Module>>>,

    // The anonymous children of this node. Anonymous children are pseudo-
    // modules that are implicitly created around items contained within
    // blocks.
    //
    // For example, if we have this:
    //
    //  fn f() {
    //      fn g() {
    //          ...
    //      }
    //  }
    //
    // There will be an anonymous module created around `g` with the ID of the
    // entry block for `f`.
    anonymous_children: RefCell<NodeMap<Rc<Module>>>,

    // The status of resolving each import in this module.
    import_resolutions: RefCell<HashMap<Name, ImportResolution>>,

    // The number of unresolved globs that this module exports.
    glob_count: Cell<uint>,

    // The index of the import we're resolving.
    resolved_import_count: Cell<uint>,

    // Whether this module is populated. If not populated, any attempt to
    // access the children must be preceded with a
    // `populate_module_if_necessary` call.
    populated: Cell<bool>,
}

impl Module {
    fn new(parent_link: ParentLink,
           def_id: Option<DefId>,
           kind: ModuleKind,
           external: bool,
           is_public: bool)
           -> Module {
        Module {
            parent_link: parent_link,
            def_id: Cell::new(def_id),
            kind: Cell::new(kind),
            is_public: is_public,
            children: RefCell::new(HashMap::new()),
            imports: RefCell::new(Vec::new()),
            external_module_children: RefCell::new(HashMap::new()),
            anonymous_children: RefCell::new(NodeMap()),
            import_resolutions: RefCell::new(HashMap::new()),
            glob_count: Cell::new(0),
            resolved_import_count: Cell::new(0),
            populated: Cell::new(!external),
        }
    }

    fn all_imports_resolved(&self) -> bool {
        self.imports.borrow().len() == self.resolved_import_count.get()
    }
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}, kind: {:?}, {}",
               self.def_id,
               self.kind,
               if self.is_public { "public" } else { "private" } )
    }
}

bitflags! {
    #[derive(Debug)]
    flags DefModifiers: u8 {
        const PUBLIC            = 0b0000_0001,
        const IMPORTABLE        = 0b0000_0010,
    }
}

// Records a possibly-private type definition.
#[derive(Clone,Debug)]
struct TypeNsDef {
    modifiers: DefModifiers, // see note in ImportResolution about how to use this
    module_def: Option<Rc<Module>>,
    type_def: Option<Def>,
    type_span: Option<Span>
}

// Records a possibly-private value definition.
#[derive(Clone, Copy, Debug)]
struct ValueNsDef {
    modifiers: DefModifiers, // see note in ImportResolution about how to use this
    def: Def,
    value_span: Option<Span>,
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
#[derive(Debug)]
struct NameBindings {
    type_def: RefCell<Option<TypeNsDef>>,   //< Meaning in type namespace.
    value_def: RefCell<Option<ValueNsDef>>, //< Meaning in value namespace.
}

/// Ways in which a trait can be referenced
#[derive(Copy)]
enum TraitReferenceType {
    TraitImplementation,             // impl SomeTrait for T { ... }
    TraitDerivation,                 // trait T : SomeTrait { ... }
    TraitBoundingTypeParameter,      // fn f<T:SomeTrait>() { ... }
    TraitObject,                     // Box<for<'a> SomeTrait>
    TraitQPath,                      // <T as SomeTrait>::
}

impl NameBindings {
    fn new() -> NameBindings {
        NameBindings {
            type_def: RefCell::new(None),
            value_def: RefCell::new(None),
        }
    }

    /// Creates a new module in this set of name bindings.
    fn define_module(&self,
                     parent_link: ParentLink,
                     def_id: Option<DefId>,
                     kind: ModuleKind,
                     external: bool,
                     is_public: bool,
                     sp: Span) {
        // Merges the module with the existing type def or creates a new one.
        let modifiers = if is_public { PUBLIC } else { DefModifiers::empty() } | IMPORTABLE;
        let module_ = Rc::new(Module::new(parent_link,
                                          def_id,
                                          kind,
                                          external,
                                          is_public));
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    modifiers: modifiers,
                    module_def: Some(module_),
                    type_def: None,
                    type_span: Some(sp)
                });
            }
            Some(type_def) => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    modifiers: modifiers,
                    module_def: Some(module_),
                    type_span: Some(sp),
                    type_def: type_def.type_def
                });
            }
        }
    }

    /// Sets the kind of the module, creating a new one if necessary.
    fn set_module_kind(&self,
                       parent_link: ParentLink,
                       def_id: Option<DefId>,
                       kind: ModuleKind,
                       external: bool,
                       is_public: bool,
                       _sp: Span) {
        let modifiers = if is_public { PUBLIC } else { DefModifiers::empty() } | IMPORTABLE;
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                let module = Module::new(parent_link,
                                         def_id,
                                         kind,
                                         external,
                                         is_public);
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    modifiers: modifiers,
                    module_def: Some(Rc::new(module)),
                    type_def: None,
                    type_span: None,
                });
            }
            Some(type_def) => {
                match type_def.module_def {
                    None => {
                        let module = Module::new(parent_link,
                                                 def_id,
                                                 kind,
                                                 external,
                                                 is_public);
                        *self.type_def.borrow_mut() = Some(TypeNsDef {
                            modifiers: modifiers,
                            module_def: Some(Rc::new(module)),
                            type_def: type_def.type_def,
                            type_span: None,
                        });
                    }
                    Some(module_def) => module_def.kind.set(kind),
                }
            }
        }
    }

    /// Records a type definition.
    fn define_type(&self, def: Def, sp: Span, modifiers: DefModifiers) {
        debug!("defining type for def {:?} with modifiers {:?}", def, modifiers);
        // Merges the type with the existing type def or creates a new one.
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    module_def: None,
                    type_def: Some(def),
                    type_span: Some(sp),
                    modifiers: modifiers,
                });
            }
            Some(type_def) => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    module_def: type_def.module_def,
                    type_def: Some(def),
                    type_span: Some(sp),
                    modifiers: modifiers,
                });
            }
        }
    }

    /// Records a value definition.
    fn define_value(&self, def: Def, sp: Span, modifiers: DefModifiers) {
        debug!("defining value for def {:?} with modifiers {:?}", def, modifiers);
        *self.value_def.borrow_mut() = Some(ValueNsDef {
            def: def,
            value_span: Some(sp),
            modifiers: modifiers,
        });
    }

    /// Returns the module node if applicable.
    fn get_module_if_available(&self) -> Option<Rc<Module>> {
        match *self.type_def.borrow() {
            Some(ref type_def) => type_def.module_def.clone(),
            None => None
        }
    }

    /// Returns the module node. Panics if this node does not have a module
    /// definition.
    fn get_module(&self) -> Rc<Module> {
        match self.get_module_if_available() {
            None => {
                panic!("get_module called on a node with no module \
                       definition!")
            }
            Some(module_def) => module_def
        }
    }

    fn defined_in_namespace(&self, namespace: Namespace) -> bool {
        match namespace {
            TypeNS   => return self.type_def.borrow().is_some(),
            ValueNS  => return self.value_def.borrow().is_some()
        }
    }

    fn defined_in_public_namespace(&self, namespace: Namespace) -> bool {
        self.defined_in_namespace_with(namespace, PUBLIC)
    }

    fn defined_in_namespace_with(&self, namespace: Namespace, modifiers: DefModifiers) -> bool {
        match namespace {
            TypeNS => match *self.type_def.borrow() {
                Some(ref def) => def.modifiers.contains(modifiers), None => false
            },
            ValueNS => match *self.value_def.borrow() {
                Some(ref def) => def.modifiers.contains(modifiers), None => false
            }
        }
    }

    fn def_for_namespace(&self, namespace: Namespace) -> Option<Def> {
        match namespace {
            TypeNS => {
                match *self.type_def.borrow() {
                    None => None,
                    Some(ref type_def) => {
                        match type_def.type_def {
                            Some(type_def) => Some(type_def),
                            None => {
                                match type_def.module_def {
                                    Some(ref module) => {
                                        match module.def_id.get() {
                                            Some(did) => Some(DefMod(did)),
                                            None => None,
                                        }
                                    }
                                    None => None,
                                }
                            }
                        }
                    }
                }
            }
            ValueNS => {
                match *self.value_def.borrow() {
                    None => None,
                    Some(value_def) => Some(value_def.def)
                }
            }
        }
    }

    fn span_for_namespace(&self, namespace: Namespace) -> Option<Span> {
        if self.defined_in_namespace(namespace) {
            match namespace {
                TypeNS  => {
                    match *self.type_def.borrow() {
                        None => None,
                        Some(ref type_def) => type_def.type_span
                    }
                }
                ValueNS => {
                    match *self.value_def.borrow() {
                        None => None,
                        Some(ref value_def) => value_def.value_span
                    }
                }
            }
        } else {
            None
        }
    }
}

/// Interns the names of the primitive types.
struct PrimitiveTypeTable {
    primitive_types: HashMap<Name, PrimTy>,
}

impl PrimitiveTypeTable {
    fn new() -> PrimitiveTypeTable {
        let mut table = PrimitiveTypeTable {
            primitive_types: HashMap::new()
        };

        table.intern("bool",    TyBool);
        table.intern("char",    TyChar);
        table.intern("f32",     TyFloat(TyF32));
        table.intern("f64",     TyFloat(TyF64));
        table.intern("int",     TyInt(TyIs(true)));
        table.intern("isize",   TyInt(TyIs(false)));
        table.intern("i8",      TyInt(TyI8));
        table.intern("i16",     TyInt(TyI16));
        table.intern("i32",     TyInt(TyI32));
        table.intern("i64",     TyInt(TyI64));
        table.intern("str",     TyStr);
        table.intern("uint",    TyUint(TyUs(true)));
        table.intern("usize",   TyUint(TyUs(false)));
        table.intern("u8",      TyUint(TyU8));
        table.intern("u16",     TyUint(TyU16));
        table.intern("u32",     TyUint(TyU32));
        table.intern("u64",     TyUint(TyU64));

        table
    }

    fn intern(&mut self, string: &str, primitive_type: PrimTy) {
        self.primitive_types.insert(token::intern(string), primitive_type);
    }
}

/// The main resolver class.
struct Resolver<'a, 'tcx:'a> {
    session: &'a Session,

    ast_map: &'a ast_map::Map<'tcx>,

    graph_root: NameBindings,

    trait_item_map: FnvHashMap<(Name, DefId), TraitItemKind>,

    structs: FnvHashMap<DefId, Vec<Name>>,

    // The number of imports that are currently unresolved.
    unresolved_imports: uint,

    // The module that represents the current item scope.
    current_module: Rc<Module>,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: Vec<Rib>,

    // The current set of local scopes, for types.
    type_ribs: Vec<Rib>,

    // The current set of local scopes, for labels.
    label_ribs: Vec<Rib>,

    // The trait that the current context can refer to.
    current_trait_ref: Option<(DefId, TraitRef)>,

    // The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    // The ident for the keyword "self".
    self_name: Name,
    // The ident for the non-keyword "Self".
    type_self_name: Name,

    // The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    def_map: DefMap,
    freevars: RefCell<FreevarMap>,
    freevars_seen: RefCell<NodeMap<NodeSet>>,
    export_map: ExportMap,
    trait_map: TraitMap,
    external_exports: ExternalExports,
    last_private: LastPrivateMap,

    // Whether or not to print error messages. Can be set to true
    // when getting additional info for error message suggestions,
    // so as to avoid printing duplicate errors
    emit_errors: bool,

    make_glob_map: bool,
    // Maps imports to the names of items actually imported (this actually maps
    // all imports, but only glob imports are actually interesting).
    glob_map: GlobMap,

    used_imports: HashSet<(NodeId, Namespace)>,
    used_crates: HashSet<CrateNum>,
}

#[derive(PartialEq)]
enum FallbackChecks {
    Everything,
    OnlyTraitAndStatics
}


impl<'a, 'tcx> Resolver<'a, 'tcx> {
    fn new(session: &'a Session,
           ast_map: &'a ast_map::Map<'tcx>,
           crate_span: Span,
           make_glob_map: MakeGlobMap) -> Resolver<'a, 'tcx> {
        let graph_root = NameBindings::new();

        graph_root.define_module(NoParentLink,
                                 Some(DefId { krate: 0, node: 0 }),
                                 NormalModuleKind,
                                 false,
                                 true,
                                 crate_span);

        let current_module = graph_root.get_module();

        Resolver {
            session: session,

            ast_map: ast_map,

            // The outermost module has def ID 0; this is not reflected in the
            // AST.

            graph_root: graph_root,

            trait_item_map: FnvHashMap(),
            structs: FnvHashMap(),

            unresolved_imports: 0,

            current_module: current_module,
            value_ribs: Vec::new(),
            type_ribs: Vec::new(),
            label_ribs: Vec::new(),

            current_trait_ref: None,
            current_self_type: None,

            self_name: special_names::self_,
            type_self_name: special_names::type_self,

            primitive_type_table: PrimitiveTypeTable::new(),

            def_map: RefCell::new(NodeMap()),
            freevars: RefCell::new(NodeMap()),
            freevars_seen: RefCell::new(NodeMap()),
            export_map: NodeMap(),
            trait_map: NodeMap(),
            used_imports: HashSet::new(),
            used_crates: HashSet::new(),
            external_exports: DefIdSet(),
            last_private: NodeMap(),

            emit_errors: true,
            make_glob_map: make_glob_map == MakeGlobMap::Yes,
            glob_map: HashMap::new(),
        }
    }

    // Import resolution
    //
    // This is a fixed-point algorithm. We resolve imports until our efforts
    // are stymied by an unresolved import; then we bail out of the current
    // module and continue. We terminate successfully once no more imports
    // remain or unsuccessfully when no forward progress in resolving imports
    // is made.

    /// Resolves all imports for the crate. This method performs the fixed-
    /// point iteration.
    fn resolve_imports(&mut self) {
        let mut i = 0;
        let mut prev_unresolved_imports = 0;
        loop {
            debug!("(resolving imports) iteration {}, {} imports left",
                   i, self.unresolved_imports);

            let module_root = self.graph_root.get_module();
            self.resolve_imports_for_module_subtree(module_root.clone());

            if self.unresolved_imports == 0 {
                debug!("(resolving imports) success");
                break;
            }

            if self.unresolved_imports == prev_unresolved_imports {
                self.report_unresolved_imports(module_root);
                break;
            }

            i += 1;
            prev_unresolved_imports = self.unresolved_imports;
        }
    }

    /// Attempts to resolve imports for the given module and all of its
    /// submodules.
    fn resolve_imports_for_module_subtree(&mut self, module_: Rc<Module>) {
        debug!("(resolving imports for module subtree) resolving {}",
               self.module_to_string(&*module_));
        let orig_module = replace(&mut self.current_module, module_.clone());
        self.resolve_imports_for_module(module_.clone());
        self.current_module = orig_module;

        build_reduced_graph::populate_module_if_necessary(self, &module_);
        for (_, child_node) in &*module_.children.borrow() {
            match child_node.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.resolve_imports_for_module_subtree(child_module);
                }
            }
        }

        for (_, child_module) in &*module_.anonymous_children.borrow() {
            self.resolve_imports_for_module_subtree(child_module.clone());
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(&mut self, module: Rc<Module>) {
        if module.all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for \
                   {}",
                   self.module_to_string(&*module));
            return;
        }

        let imports = module.imports.borrow();
        let import_count = imports.len();
        while module.resolved_import_count.get() < import_count {
            let import_index = module.resolved_import_count.get();
            let import_directive = &(*imports)[import_index];
            match self.resolve_import_for_module(module.clone(),
                                                 import_directive) {
                Failed(err) => {
                    let (span, help) = match err {
                        Some((span, msg)) => (span, format!(". {}", msg)),
                        None => (import_directive.span, String::new())
                    };
                    let msg = format!("unresolved import `{}`{}",
                                      self.import_path_to_string(
                                          &import_directive.module_path[],
                                          import_directive.subclass),
                                      help);
                    self.resolve_error(span, &msg[..]);
                }
                Indeterminate => break, // Bail out. We'll come around next time.
                Success(()) => () // Good. Continue.
            }

            module.resolved_import_count
                  .set(module.resolved_import_count.get() + 1);
        }
    }

    fn names_to_string(&self, names: &[Name]) -> String {
        let mut first = true;
        let mut result = String::new();
        for name in names {
            if first {
                first = false
            } else {
                result.push_str("::")
            }
            result.push_str(&token::get_name(*name));
        };
        result
    }

    fn path_names_to_string(&self, path: &Path) -> String {
        let names: Vec<ast::Name> = path.segments
                                        .iter()
                                        .map(|seg| seg.identifier.name)
                                        .collect();
        self.names_to_string(&names[..])
    }

    fn import_directive_subclass_to_string(&mut self,
                                        subclass: ImportDirectiveSubclass)
                                        -> String {
        match subclass {
            SingleImport(_, source) => {
                token::get_name(source).to_string()
            }
            GlobImport => "*".to_string()
        }
    }

    fn import_path_to_string(&mut self,
                          names: &[Name],
                          subclass: ImportDirectiveSubclass)
                          -> String {
        if names.is_empty() {
            self.import_directive_subclass_to_string(subclass)
        } else {
            (format!("{}::{}",
                     self.names_to_string(names),
                     self.import_directive_subclass_to_string(
                         subclass))).to_string()
        }
    }

    #[inline]
    fn record_import_use(&mut self, import_id: NodeId, name: Name) {
        if !self.make_glob_map {
            return;
        }
        if self.glob_map.contains_key(&import_id) {
            self.glob_map[import_id].insert(name);
            return;
        }

        let mut new_set = HashSet::new();
        new_set.insert(name);
        self.glob_map.insert(import_id, new_set);
    }

    fn get_trait_name(&self, did: DefId) -> Name {
        if did.krate == ast::LOCAL_CRATE {
            self.ast_map.expect_item(did.node).ident.name
        } else {
            csearch::get_trait_name(&self.session.cstore, did)
        }
    }

    /// Attempts to resolve the given import. The return value indicates
    /// failure if we're certain the name does not exist, indeterminate if we
    /// don't know whether the name exists at the moment due to other
    /// currently-unresolved imports, or success if we know the name exists.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import_for_module(&mut self,
                                 module_: Rc<Module>,
                                 import_directive: &ImportDirective)
                                 -> ResolveResult<()> {
        let mut resolution_result = Failed(None);
        let module_path = &import_directive.module_path;

        debug!("(resolving import for module) resolving import `{}::...` in `{}`",
               self.names_to_string(&module_path[..]),
               self.module_to_string(&*module_));

        // First, resolve the module path for the directive, if necessary.
        let container = if module_path.len() == 0 {
            // Use the crate root.
            Some((self.graph_root.get_module(), LastMod(AllPublic)))
        } else {
            match self.resolve_module_path(module_.clone(),
                                           &module_path[..],
                                           DontUseLexicalScope,
                                           import_directive.span,
                                           ImportSearch) {
                Failed(err) => {
                    resolution_result = Failed(err);
                    None
                },
                Indeterminate => {
                    resolution_result = Indeterminate;
                    None
                }
                Success(container) => Some(container),
            }
        };

        match container {
            None => {}
            Some((containing_module, lp)) => {
                // We found the module that the target is contained
                // within. Attempt to resolve the import within it.

                match import_directive.subclass {
                    SingleImport(target, source) => {
                        resolution_result =
                            self.resolve_single_import(&*module_,
                                                       containing_module,
                                                       target,
                                                       source,
                                                       import_directive,
                                                       lp);
                    }
                    GlobImport => {
                        resolution_result =
                            self.resolve_glob_import(&*module_,
                                                     containing_module,
                                                     import_directive,
                                                     lp);
                    }
                }
            }
        }

        // Decrement the count of unresolved imports.
        match resolution_result {
            Success(()) => {
                assert!(self.unresolved_imports >= 1);
                self.unresolved_imports -= 1;
            }
            _ => {
                // Nothing to do here; just return the error.
            }
        }

        // Decrement the count of unresolved globs if necessary. But only if
        // the resolution result is indeterminate -- otherwise we'll stop
        // processing imports here. (See the loop in
        // resolve_imports_for_module.)

        if !resolution_result.indeterminate() {
            match import_directive.subclass {
                GlobImport => {
                    assert!(module_.glob_count.get() >= 1);
                    module_.glob_count.set(module_.glob_count.get() - 1);
                }
                SingleImport(..) => {
                    // Ignore.
                }
            }
        }

        return resolution_result;
    }

    fn create_name_bindings_from_module(module: Rc<Module>) -> NameBindings {
        NameBindings {
            type_def: RefCell::new(Some(TypeNsDef {
                modifiers: IMPORTABLE,
                module_def: Some(module),
                type_def: None,
                type_span: None
            })),
            value_def: RefCell::new(None),
        }
    }

    fn resolve_single_import(&mut self,
                             module_: &Module,
                             containing_module: Rc<Module>,
                             target: Name,
                             source: Name,
                             directive: &ImportDirective,
                             lp: LastPrivate)
                                 -> ResolveResult<()> {
        debug!("(resolving single import) resolving `{}` = `{}::{}` from \
                `{}` id {}, last private {:?}",
               token::get_name(target),
               self.module_to_string(&*containing_module),
               token::get_name(source),
               self.module_to_string(module_),
               directive.id,
               lp);

        let lp = match lp {
            LastMod(lp) => lp,
            LastImport {..} => {
                self.session
                    .span_bug(directive.span,
                              "not expecting Import here, must be LastMod")
            }
        };

        // We need to resolve both namespaces for this to succeed.
        //

        let mut value_result = UnknownResult;
        let mut type_result = UnknownResult;

        // Search for direct children of the containing module.
        build_reduced_graph::populate_module_if_necessary(self, &containing_module);

        match containing_module.children.borrow().get(&source) {
            None => {
                // Continue.
            }
            Some(ref child_name_bindings) => {
                if child_name_bindings.defined_in_namespace(ValueNS) {
                    debug!("(resolving single import) found value binding");
                    value_result = BoundResult(containing_module.clone(),
                                               (*child_name_bindings).clone());
                }
                if child_name_bindings.defined_in_namespace(TypeNS) {
                    debug!("(resolving single import) found type binding");
                    type_result = BoundResult(containing_module.clone(),
                                              (*child_name_bindings).clone());
                }
            }
        }

        // Unless we managed to find a result in both namespaces (unlikely),
        // search imports as well.
        let mut value_used_reexport = false;
        let mut type_used_reexport = false;
        match (value_result.clone(), type_result.clone()) {
            (BoundResult(..), BoundResult(..)) => {} // Continue.
            _ => {
                // If there is an unresolved glob at this point in the
                // containing module, bail out. We don't know enough to be
                // able to resolve this import.

                if containing_module.glob_count.get() > 0 {
                    debug!("(resolving single import) unresolved glob; \
                            bailing out");
                    return Indeterminate;
                }

                // Now search the exported imports within the containing module.
                match containing_module.import_resolutions.borrow().get(&source) {
                    None => {
                        debug!("(resolving single import) no import");
                        // The containing module definitely doesn't have an
                        // exported import with the name in question. We can
                        // therefore accurately report that the names are
                        // unbound.

                        if value_result.is_unknown() {
                            value_result = UnboundResult;
                        }
                        if type_result.is_unknown() {
                            type_result = UnboundResult;
                        }
                    }
                    Some(import_resolution)
                            if import_resolution.outstanding_references == 0 => {

                        fn get_binding(this: &mut Resolver,
                                       import_resolution: &ImportResolution,
                                       namespace: Namespace,
                                       source: &Name)
                                    -> NamespaceResult {

                            // Import resolutions must be declared with "pub"
                            // in order to be exported.
                            if !import_resolution.is_public {
                                return UnboundResult;
                            }

                            match import_resolution.
                                    target_for_namespace(namespace) {
                                None => {
                                    return UnboundResult;
                                }
                                Some(Target {
                                    target_module,
                                    bindings,
                                    shadowable: _
                                }) => {
                                    debug!("(resolving single import) found \
                                            import in ns {:?}", namespace);
                                    let id = import_resolution.id(namespace);
                                    // track used imports and extern crates as well
                                    this.used_imports.insert((id, namespace));
                                    this.record_import_use(id, *source);
                                    match target_module.def_id.get() {
                                        Some(DefId{krate: kid, ..}) => {
                                            this.used_crates.insert(kid);
                                        },
                                        _ => {}
                                    }
                                    return BoundResult(target_module, bindings);
                                }
                            }
                        }

                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.
                        if value_result.is_unknown() {
                            value_result = get_binding(self,
                                                       import_resolution,
                                                       ValueNS,
                                                       &source);
                            value_used_reexport = import_resolution.is_public;
                        }
                        if type_result.is_unknown() {
                            type_result = get_binding(self,
                                                      import_resolution,
                                                      TypeNS,
                                                      &source);
                            type_used_reexport = import_resolution.is_public;
                        }

                    }
                    Some(_) => {
                        // If containing_module is the same module whose import we are resolving
                        // and there it has an unresolved import with the same name as `source`,
                        // then the user is actually trying to import an item that is declared
                        // in the same scope
                        //
                        // e.g
                        // use self::submodule;
                        // pub mod submodule;
                        //
                        // In this case we continue as if we resolved the import and let the
                        // check_for_conflicts_between_imports_and_items call below handle
                        // the conflict
                        match (module_.def_id.get(),  containing_module.def_id.get()) {
                            (Some(id1), Some(id2)) if id1 == id2  => {
                                if value_result.is_unknown() {
                                    value_result = UnboundResult;
                                }
                                if type_result.is_unknown() {
                                    type_result = UnboundResult;
                                }
                            }
                            _ =>  {
                                // The import is unresolved. Bail out.
                                debug!("(resolving single import) unresolved import; \
                                        bailing out");
                                return Indeterminate;
                            }
                        }
                    }
                }
            }
        }

        // If we didn't find a result in the type namespace, search the
        // external modules.
        let mut value_used_public = false;
        let mut type_used_public = false;
        match type_result {
            BoundResult(..) => {}
            _ => {
                match containing_module.external_module_children.borrow_mut()
                                       .get(&source).cloned() {
                    None => {} // Continue.
                    Some(module) => {
                        debug!("(resolving single import) found external \
                                module");
                        // track the module as used.
                        match module.def_id.get() {
                            Some(DefId{krate: kid, ..}) => { self.used_crates.insert(kid); },
                            _ => {}
                        }
                        let name_bindings =
                            Rc::new(Resolver::create_name_bindings_from_module(
                                module));
                        type_result = BoundResult(containing_module.clone(),
                                                  name_bindings);
                        type_used_public = true;
                    }
                }
            }
        }

        // We've successfully resolved the import. Write the results in.
        let mut import_resolutions = module_.import_resolutions.borrow_mut();
        let import_resolution = &mut (*import_resolutions)[target];
        {
            let mut check_and_write_import = |namespace, result: &_, used_public: &mut bool| {
                let namespace_name = match namespace {
                    TypeNS => "type",
                    ValueNS => "value",
                };

                match *result {
                    BoundResult(ref target_module, ref name_bindings) => {
                        debug!("(resolving single import) found {:?} target: {:?}",
                               namespace_name,
                               name_bindings.def_for_namespace(namespace));
                        self.check_for_conflicting_import(
                            &import_resolution.target_for_namespace(namespace),
                            directive.span,
                            target,
                            namespace);

                        self.check_that_import_is_importable(
                            &**name_bindings,
                            directive.span,
                            target,
                            namespace);

                        let target = Some(Target::new(target_module.clone(),
                                                      name_bindings.clone(),
                                                      directive.shadowable));
                        import_resolution.set_target_and_id(namespace, target, directive.id);
                        import_resolution.is_public = directive.is_public;
                        *used_public = name_bindings.defined_in_public_namespace(namespace);
                    }
                    UnboundResult => { /* Continue. */ }
                    UnknownResult => {
                        panic!("{:?} result should be known at this point", namespace_name);
                    }
                }
            };
            check_and_write_import(ValueNS, &value_result, &mut value_used_public);
            check_and_write_import(TypeNS, &type_result, &mut type_used_public);
        }

        self.check_for_conflicts_between_imports_and_items(
            module_,
            import_resolution,
            directive.span,
            target);

        if value_result.is_unbound() && type_result.is_unbound() {
            let msg = format!("There is no `{}` in `{}`",
                              token::get_name(source),
                              self.module_to_string(&*containing_module));
            return Failed(Some((directive.span, msg)));
        }
        let value_used_public = value_used_reexport || value_used_public;
        let type_used_public = type_used_reexport || type_used_public;

        assert!(import_resolution.outstanding_references >= 1);
        import_resolution.outstanding_references -= 1;

        // record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        let value_private = match import_resolution.value_target {
            Some(ref target) => {
                let def = target.bindings.def_for_namespace(ValueNS).unwrap();
                self.def_map.borrow_mut().insert(directive.id, def);
                let did = def.def_id();
                if value_used_public {Some(lp)} else {Some(DependsOn(did))}
            },
            // AllPublic here and below is a dummy value, it should never be used because
            // _exists is false.
            None => None,
        };
        let type_private = match import_resolution.type_target {
            Some(ref target) => {
                let def = target.bindings.def_for_namespace(TypeNS).unwrap();
                self.def_map.borrow_mut().insert(directive.id, def);
                let did = def.def_id();
                if type_used_public {Some(lp)} else {Some(DependsOn(did))}
            },
            None => None,
        };

        self.last_private.insert(directive.id, LastImport{value_priv: value_private,
                                                          value_used: Used,
                                                          type_priv: type_private,
                                                          type_used: Used});

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    // Resolves a glob import. Note that this function cannot fail; it either
    // succeeds or bails out (as importing * from an empty module or a module
    // that exports nothing is valid). containing_module is the module we are
    // actually importing, i.e., `foo` in `use foo::*`.
    fn resolve_glob_import(&mut self,
                           module_: &Module,
                           containing_module: Rc<Module>,
                           import_directive: &ImportDirective,
                           lp: LastPrivate)
                           -> ResolveResult<()> {
        let id = import_directive.id;
        let is_public = import_directive.is_public;

        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.
        debug!("(resolving glob import) resolving glob import {}", id);

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).
        if !(*containing_module).all_imports_resolved() {
            debug!("(resolving glob import) target module has unresolved \
                    imports; bailing out");
            return Indeterminate;
        }

        assert_eq!(containing_module.glob_count.get(), 0);

        // Add all resolved imports from the containing module.
        let import_resolutions = containing_module.import_resolutions.borrow();
        for (ident, target_import_resolution) in &*import_resolutions {
            debug!("(resolving glob import) writing module resolution \
                    {} into `{}`",
                   token::get_name(*ident),
                   self.module_to_string(module_));

            if !target_import_resolution.is_public {
                debug!("(resolving glob import) nevermind, just kidding");
                continue
            }

            // Here we merge two import resolutions.
            let mut import_resolutions = module_.import_resolutions.borrow_mut();
            match import_resolutions.get_mut(ident) {
                Some(dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match target_import_resolution.value_target {
                        None => {
                            // Continue.
                        }
                        Some(ref value_target) => {
                            self.check_for_conflicting_import(&dest_import_resolution.value_target,
                                                              import_directive.span,
                                                              *ident,
                                                              ValueNS);
                            dest_import_resolution.value_target = Some(value_target.clone());
                        }
                    }
                    match target_import_resolution.type_target {
                        None => {
                            // Continue.
                        }
                        Some(ref type_target) => {
                            self.check_for_conflicting_import(&dest_import_resolution.type_target,
                                                              import_directive.span,
                                                              *ident,
                                                              TypeNS);
                            dest_import_resolution.type_target = Some(type_target.clone());
                        }
                    }
                    dest_import_resolution.is_public = is_public;
                    continue;
                }
                None => {}
            }

            // Simple: just copy the old import resolution.
            let mut new_import_resolution = ImportResolution::new(id, is_public);
            new_import_resolution.value_target =
                target_import_resolution.value_target.clone();
            new_import_resolution.type_target =
                target_import_resolution.type_target.clone();

            import_resolutions.insert(*ident, new_import_resolution);
        }

        // Add all children from the containing module.
        build_reduced_graph::populate_module_if_necessary(self, &containing_module);

        for (&name, name_bindings) in &*containing_module.children.borrow() {
            self.merge_import_resolution(module_,
                                         containing_module.clone(),
                                         import_directive,
                                         name,
                                         name_bindings.clone());

        }

        // Add external module children from the containing module.
        for (&name, module) in &*containing_module.external_module_children.borrow() {
            let name_bindings =
                Rc::new(Resolver::create_name_bindings_from_module(module.clone()));
            self.merge_import_resolution(module_,
                                         containing_module.clone(),
                                         import_directive,
                                         name,
                                         name_bindings);
        }

        // Record the destination of this import
        match containing_module.def_id.get() {
            Some(did) => {
                self.def_map.borrow_mut().insert(id, DefMod(did));
                self.last_private.insert(id, lp);
            }
            None => {}
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    fn merge_import_resolution(&mut self,
                               module_: &Module,
                               containing_module: Rc<Module>,
                               import_directive: &ImportDirective,
                               name: Name,
                               name_bindings: Rc<NameBindings>) {
        let id = import_directive.id;
        let is_public = import_directive.is_public;

        let mut import_resolutions = module_.import_resolutions.borrow_mut();
        let dest_import_resolution = import_resolutions.entry(name).get().unwrap_or_else(
            |vacant_entry| {
                // Create a new import resolution from this child.
                vacant_entry.insert(ImportResolution::new(id, is_public))
            });

        debug!("(resolving glob import) writing resolution `{}` in `{}` \
               to `{}`",
               &token::get_name(name),
               self.module_to_string(&*containing_module),
               self.module_to_string(module_));

        // Merge the child item into the import resolution.
        {
            let mut merge_child_item = |namespace| {
                if name_bindings.defined_in_namespace_with(namespace, IMPORTABLE | PUBLIC) {
                    let namespace_name = match namespace {
                        TypeNS => "type",
                        ValueNS => "value",
                    };
                    debug!("(resolving glob import) ... for {} target", namespace_name);
                    if dest_import_resolution.shadowable(namespace) == Shadowable::Never {
                        let msg = format!("a {} named `{}` has already been imported \
                                           in this module",
                                          namespace_name,
                                          &token::get_name(name));
                        span_err!(self.session, import_directive.span, E0251, "{}", msg);
                    } else {
                        let target = Target::new(containing_module.clone(),
                                                 name_bindings.clone(),
                                                 import_directive.shadowable);
                        dest_import_resolution.set_target_and_id(namespace,
                                                                 Some(target),
                                                                 id);
                    }
                }
            };
            merge_child_item(ValueNS);
            merge_child_item(TypeNS);
        }

        dest_import_resolution.is_public = is_public;

        self.check_for_conflicts_between_imports_and_items(
            module_,
            dest_import_resolution,
            import_directive.span,
            name);
    }

    /// Checks that imported names and items don't have the same name.
    fn check_for_conflicting_import(&mut self,
                                    target: &Option<Target>,
                                    import_span: Span,
                                    name: Name,
                                    namespace: Namespace) {
        debug!("check_for_conflicting_import: {}; target exists: {}",
               &token::get_name(name),
               target.is_some());

        match *target {
            Some(ref target) if target.shadowable != Shadowable::Always => {
                let msg = format!("a {} named `{}` has already been imported \
                                   in this module",
                                  match namespace {
                                    TypeNS => "type",
                                    ValueNS => "value",
                                  },
                                  &token::get_name(name));
                span_err!(self.session, import_span, E0252, "{}", &msg[..]);
            }
            Some(_) | None => {}
        }
    }

    /// Checks that an import is actually importable
    fn check_that_import_is_importable(&mut self,
                                       name_bindings: &NameBindings,
                                       import_span: Span,
                                       name: Name,
                                       namespace: Namespace) {
        if !name_bindings.defined_in_namespace_with(namespace, IMPORTABLE) {
            let msg = format!("`{}` is not directly importable",
                              token::get_name(name));
            span_err!(self.session, import_span, E0253, "{}", &msg[..]);
        }
    }

    /// Checks that imported names and items don't have the same name.
    fn check_for_conflicts_between_imports_and_items(&mut self,
                                                     module: &Module,
                                                     import_resolution:
                                                     &ImportResolution,
                                                     import_span: Span,
                                                     name: Name) {
        // First, check for conflicts between imports and `extern crate`s.
        if module.external_module_children
                 .borrow()
                 .contains_key(&name) {
            match import_resolution.type_target {
                Some(ref target) if target.shadowable != Shadowable::Always => {
                    let msg = format!("import `{0}` conflicts with imported \
                                       crate in this module \
                                       (maybe you meant `use {0}::*`?)",
                                      &token::get_name(name));
                    span_err!(self.session, import_span, E0254, "{}", &msg[..]);
                }
                Some(_) | None => {}
            }
        }

        // Check for item conflicts.
        let children = module.children.borrow();
        let name_bindings = match children.get(&name) {
            None => {
                // There can't be any conflicts.
                return
            }
            Some(ref name_bindings) => (*name_bindings).clone(),
        };

        match import_resolution.value_target {
            Some(ref target) if target.shadowable != Shadowable::Always => {
                if let Some(ref value) = *name_bindings.value_def.borrow() {
                    let msg = format!("import `{}` conflicts with value \
                                       in this module",
                                      &token::get_name(name));
                    span_err!(self.session, import_span, E0255, "{}", &msg[..]);
                    if let Some(span) = value.value_span {
                        self.session.span_note(span,
                                               "conflicting value here");
                    }
                }
            }
            Some(_) | None => {}
        }

        match import_resolution.type_target {
            Some(ref target) if target.shadowable != Shadowable::Always => {
                if let Some(ref ty) = *name_bindings.type_def.borrow() {
                    match ty.module_def {
                        None => {
                            let msg = format!("import `{}` conflicts with type in \
                                               this module",
                                              &token::get_name(name));
                            span_err!(self.session, import_span, E0256, "{}", &msg[..]);
                            if let Some(span) = ty.type_span {
                                self.session.span_note(span,
                                                       "note conflicting type here")
                            }
                        }
                        Some(ref module_def) => {
                            match module_def.kind.get() {
                                ImplModuleKind => {
                                    if let Some(span) = ty.type_span {
                                        let msg = format!("inherent implementations \
                                                           are only allowed on types \
                                                           defined in the current module");
                                        span_err!(self.session, span, E0257, "{}", &msg[..]);
                                        self.session.span_note(import_span,
                                                               "import from other module here")
                                    }
                                }
                                _ => {
                                    let msg = format!("import `{}` conflicts with existing \
                                                       submodule",
                                                      &token::get_name(name));
                                    span_err!(self.session, import_span, E0258, "{}", &msg[..]);
                                    if let Some(span) = ty.type_span {
                                        self.session.span_note(span,
                                                               "note conflicting module here")
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Some(_) | None => {}
        }
    }

    /// Checks that the names of external crates don't collide with other
    /// external crates.
    fn check_for_conflicts_between_external_crates(&self,
                                                   module: &Module,
                                                   name: Name,
                                                   span: Span) {
        if module.external_module_children.borrow().contains_key(&name) {
                span_err!(self.session, span, E0259,
                          "an external crate named `{}` has already \
                                   been imported into this module",
                                  &token::get_name(name));
        }
    }

    /// Checks that the names of items don't collide with external crates.
    fn check_for_conflicts_between_external_crates_and_items(&self,
                                                             module: &Module,
                                                             name: Name,
                                                             span: Span) {
        if module.external_module_children.borrow().contains_key(&name) {
                span_err!(self.session, span, E0260,
                          "the name `{}` conflicts with an external \
                                   crate that has been imported into this \
                                   module",
                                  &token::get_name(name));
        }
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(&mut self,
                                     module_: Rc<Module>,
                                     module_path: &[Name],
                                     index: uint,
                                     span: Span,
                                     name_search_type: NameSearchType,
                                     lp: LastPrivate)
                                -> ResolveResult<(Rc<Module>, LastPrivate)> {
        fn search_parent_externals(needle: Name, module: &Rc<Module>)
                                -> Option<Rc<Module>> {
            match module.external_module_children.borrow().get(&needle) {
                Some(_) => Some(module.clone()),
                None => match module.parent_link {
                    ModuleParentLink(ref parent, _) => {
                        search_parent_externals(needle, &parent.upgrade().unwrap())
                    }
                   _ => None
                }
            }
        }

        let mut search_module = module_;
        let mut index = index;
        let module_path_len = module_path.len();
        let mut closest_private = lp;

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.
        while index < module_path_len {
            let name = module_path[index];
            match self.resolve_name_in_module(search_module.clone(),
                                              name,
                                              TypeNS,
                                              name_search_type,
                                              false) {
                Failed(None) => {
                    let segment_name = token::get_name(name);
                    let module_name = self.module_to_string(&*search_module);
                    let mut span = span;
                    let msg = if "???" == &module_name[..] {
                        span.hi = span.lo + Pos::from_usize(segment_name.len());

                        match search_parent_externals(name,
                                                     &self.current_module) {
                            Some(module) => {
                                let path_str = self.names_to_string(module_path);
                                let target_mod_str = self.module_to_string(&*module);
                                let current_mod_str =
                                    self.module_to_string(&*self.current_module);

                                let prefix = if target_mod_str == current_mod_str {
                                    "self::".to_string()
                                } else {
                                    format!("{}::", target_mod_str)
                                };

                                format!("Did you mean `{}{}`?", prefix, path_str)
                            },
                            None => format!("Maybe a missing `extern crate {}`?",
                                            segment_name),
                        }
                    } else {
                        format!("Could not find `{}` in `{}`",
                                segment_name,
                                module_name)
                    };

                    return Failed(Some((span, msg)));
                }
                Failed(err) => return Failed(err),
                Indeterminate => {
                    debug!("(resolving module path for import) module \
                            resolution is indeterminate: {}",
                            token::get_name(name));
                    return Indeterminate;
                }
                Success((target, used_proxy)) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    match *target.bindings.type_def.borrow() {
                        Some(ref type_def) => {
                            match type_def.module_def {
                                None => {
                                    let msg = format!("Not a module `{}`",
                                                        token::get_name(name));

                                    return Failed(Some((span, msg)));
                                }
                                Some(ref module_def) => {
                                    search_module = module_def.clone();

                                    // track extern crates for unused_extern_crate lint
                                    if let Some(did) = module_def.def_id.get() {
                                        self.used_crates.insert(did.krate);
                                    }

                                    // Keep track of the closest
                                    // private module used when
                                    // resolving this import chain.
                                    if !used_proxy && !search_module.is_public {
                                        if let Some(did) = search_module.def_id.get() {
                                            closest_private = LastMod(DependsOn(did));
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            // There are no type bindings at all.
                            let msg = format!("Not a module `{}`",
                                              token::get_name(name));
                            return Failed(Some((span, msg)));
                        }
                    }
                }
            }

            index += 1;
        }

        return Success((search_module, closest_private));
    }

    /// Attempts to resolve the module part of an import directive or path
    /// rooted at the given module.
    ///
    /// On success, returns the resolved module, and the closest *private*
    /// module found to the destination when resolving this path.
    fn resolve_module_path(&mut self,
                           module_: Rc<Module>,
                           module_path: &[Name],
                           use_lexical_scope: UseLexicalScopeFlag,
                           span: Span,
                           name_search_type: NameSearchType)
                           -> ResolveResult<(Rc<Module>, LastPrivate)> {
        let module_path_len = module_path.len();
        assert!(module_path_len > 0);

        debug!("(resolving module path for import) processing `{}` rooted at `{}`",
               self.names_to_string(module_path),
               self.module_to_string(&*module_));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_.clone(),
                                                              module_path);

        let search_module;
        let start_index;
        let last_private;
        match module_prefix_result {
            Failed(None) => {
                let mpath = self.names_to_string(module_path);
                let mpath = &mpath[..];
                match mpath.rfind(':') {
                    Some(idx) => {
                        let msg = format!("Could not find `{}` in `{}`",
                                            // idx +- 1 to account for the
                                            // colons on either side
                                            &mpath[idx + 1..],
                                            &mpath[..idx - 1]);
                        return Failed(Some((span, msg)));
                    },
                    None => {
                        return Failed(None)
                    }
                }
            }
            Failed(err) => return Failed(err),
            Indeterminate => {
                debug!("(resolving module path for import) indeterminate; \
                        bailing");
                return Indeterminate;
            }
            Success(NoPrefixFound) => {
                // There was no prefix, so we're considering the first element
                // of the path. How we handle this depends on whether we were
                // instructed to use lexical scope or not.
                match use_lexical_scope {
                    DontUseLexicalScope => {
                        // This is a crate-relative path. We will start the
                        // resolution process at index zero.
                        search_module = self.graph_root.get_module();
                        start_index = 0;
                        last_private = LastMod(AllPublic);
                    }
                    UseLexicalScope => {
                        // This is not a crate-relative path. We resolve the
                        // first component of the path in the current lexical
                        // scope and then proceed to resolve below that.
                        match self.resolve_module_in_lexical_scope(module_,
                                                                   module_path[0]) {
                            Failed(err) => return Failed(err),
                            Indeterminate => {
                                debug!("(resolving module path for import) \
                                        indeterminate; bailing");
                                return Indeterminate;
                            }
                            Success(containing_module) => {
                                search_module = containing_module;
                                start_index = 1;
                                last_private = LastMod(AllPublic);
                            }
                        }
                    }
                }
            }
            Success(PrefixFound(ref containing_module, index)) => {
                search_module = containing_module.clone();
                start_index = index;
                last_private = LastMod(DependsOn(containing_module.def_id
                                                                  .get()
                                                                  .unwrap()));
            }
        }

        self.resolve_module_path_from_root(search_module,
                                           module_path,
                                           start_index,
                                           span,
                                           name_search_type,
                                           last_private)
    }

    /// Invariant: This must only be called during main resolution, not during
    /// import resolution.
    fn resolve_item_in_lexical_scope(&mut self,
                                     module_: Rc<Module>,
                                     name: Name,
                                     namespace: Namespace)
                                    -> ResolveResult<(Target, bool)> {
        debug!("(resolving item in lexical scope) resolving `{}` in \
                namespace {:?} in `{}`",
               token::get_name(name),
               namespace,
               self.module_to_string(&*module_));

        // The current module node is handled specially. First, check for
        // its immediate children.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        match module_.children.borrow().get(&name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                debug!("top name bindings succeeded");
                return Success((Target::new(module_.clone(),
                                            name_bindings.clone(),
                                            Shadowable::Never),
                               false));
            }
            Some(_) | None => { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.
        if let Some(import_resolution) = module_.import_resolutions.borrow().get(&name) {
            match (*import_resolution).target_for_namespace(namespace) {
                None => {
                    // Not found; continue.
                    debug!("(resolving item in lexical scope) found \
                            import resolution, but not in namespace {:?}",
                           namespace);
                }
                Some(target) => {
                    debug!("(resolving item in lexical scope) using \
                            import resolution");
                    // track used imports and extern crates as well
                    let id = import_resolution.id(namespace);
                    self.used_imports.insert((id, namespace));
                    self.record_import_use(id, name);
                    if let Some(DefId{krate: kid, ..}) = target.target_module.def_id.get() {
                         self.used_crates.insert(kid);
                    }
                    return Success((target, false));
                }
            }
        }

        // Search for external modules.
        if namespace == TypeNS {
            // FIXME (21114): In principle unclear `child` *has* to be lifted.
            let child = module_.external_module_children.borrow().get(&name).cloned();
            if let Some(module) = child {
                let name_bindings =
                    Rc::new(Resolver::create_name_bindings_from_module(module));
                debug!("lower name bindings succeeded");
                return Success((Target::new(module_,
                                            name_bindings,
                                            Shadowable::Never),
                                false));
            }
        }

        // Finally, proceed up the scope chain looking for parent modules.
        let mut search_module = module_;
        loop {
            // Go to the next parent.
            match search_module.parent_link.clone() {
                NoParentLink => {
                    // No more parents. This module was unresolved.
                    debug!("(resolving item in lexical scope) unresolved \
                            module");
                    return Failed(None);
                }
                ModuleParentLink(parent_module_node, _) => {
                    match search_module.kind.get() {
                        NormalModuleKind => {
                            // We stop the search here.
                            debug!("(resolving item in lexical \
                                    scope) unresolved module: not \
                                    searching through module \
                                    parents");
                            return Failed(None);
                        }
                        TraitModuleKind |
                        ImplModuleKind |
                        EnumModuleKind |
                        TypeModuleKind |
                        AnonymousModuleKind => {
                            search_module = parent_module_node.upgrade().unwrap();
                        }
                    }
                }
                BlockParentLink(ref parent_module_node, _) => {
                    search_module = parent_module_node.upgrade().unwrap();
                }
            }

            // Resolve the name in the parent module.
            match self.resolve_name_in_module(search_module.clone(),
                                              name,
                                              namespace,
                                              PathSearch,
                                              true) {
                Failed(Some((span, msg))) =>
                    self.resolve_error(span, &format!("failed to resolve. {}",
                                                     msg)[]),
                Failed(None) => (), // Continue up the search chain.
                Indeterminate => {
                    // We couldn't see through the higher scope because of an
                    // unresolved import higher up. Bail.

                    debug!("(resolving item in lexical scope) indeterminate \
                            higher scope; bailing");
                    return Indeterminate;
                }
                Success((target, used_reexport)) => {
                    // We found the module.
                    debug!("(resolving item in lexical scope) found name \
                            in module, done");
                    return Success((target, used_reexport));
                }
            }
        }
    }

    /// Resolves a module name in the current lexical scope.
    fn resolve_module_in_lexical_scope(&mut self,
                                       module_: Rc<Module>,
                                       name: Name)
                                -> ResolveResult<Rc<Module>> {
        // If this module is an anonymous module, resolve the item in the
        // lexical scope. Otherwise, resolve the item from the crate root.
        let resolve_result = self.resolve_item_in_lexical_scope(module_, name, TypeNS);
        match resolve_result {
            Success((target, _)) => {
                let bindings = &*target.bindings;
                match *bindings.type_def.borrow() {
                    Some(ref type_def) => {
                        match type_def.module_def {
                            None => {
                                debug!("!!! (resolving module in lexical \
                                        scope) module wasn't actually a \
                                        module!");
                                return Failed(None);
                            }
                            Some(ref module_def) => {
                                return Success(module_def.clone());
                            }
                        }
                    }
                    None => {
                        debug!("!!! (resolving module in lexical scope) module
                                wasn't actually a module!");
                        return Failed(None);
                    }
                }
            }
            Indeterminate => {
                debug!("(resolving module in lexical scope) indeterminate; \
                        bailing");
                return Indeterminate;
            }
            Failed(err) => {
                debug!("(resolving module in lexical scope) failed to resolve");
                return Failed(err);
            }
        }
    }

    /// Returns the nearest normal module parent of the given module.
    fn get_nearest_normal_module_parent(&mut self, module_: Rc<Module>)
                                            -> Option<Rc<Module>> {
        let mut module_ = module_;
        loop {
            match module_.parent_link.clone() {
                NoParentLink => return None,
                ModuleParentLink(new_module, _) |
                BlockParentLink(new_module, _) => {
                    let new_module = new_module.upgrade().unwrap();
                    match new_module.kind.get() {
                        NormalModuleKind => return Some(new_module),
                        TraitModuleKind |
                        ImplModuleKind |
                        EnumModuleKind |
                        TypeModuleKind |
                        AnonymousModuleKind => module_ = new_module,
                    }
                }
            }
        }
    }

    /// Returns the nearest normal module parent of the given module, or the
    /// module itself if it is a normal module.
    fn get_nearest_normal_module_parent_or_self(&mut self, module_: Rc<Module>)
                                                -> Rc<Module> {
        match module_.kind.get() {
            NormalModuleKind => return module_,
            TraitModuleKind |
            ImplModuleKind |
            EnumModuleKind |
            TypeModuleKind |
            AnonymousModuleKind => {
                match self.get_nearest_normal_module_parent(module_.clone()) {
                    None => module_,
                    Some(new_module) => new_module
                }
            }
        }
    }

    /// Resolves a "module prefix". A module prefix is one or both of (a) `self::`;
    /// (b) some chain of `super::`.
    /// grammar: (SELF MOD_SEP ) ? (SUPER MOD_SEP) *
    fn resolve_module_prefix(&mut self,
                             module_: Rc<Module>,
                             module_path: &[Name])
                                 -> ResolveResult<ModulePrefixResult> {
        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut containing_module;
        let mut i;
        let first_module_path_string = token::get_name(module_path[0]);
        if "self" == &first_module_path_string[..] {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 1;
        } else if "super" == &first_module_path_string[..] {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 0;  // We'll handle `super` below.
        } else {
            return Success(NoPrefixFound);
        }

        // Now loop through all the `super`s we find.
        while i < module_path.len() {
            let string = token::get_name(module_path[i]);
            if "super" != &string[..] {
                break
            }
            debug!("(resolving module prefix) resolving `super` at {}",
                   self.module_to_string(&*containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => return Failed(None),
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at {}",
               self.module_to_string(&*containing_module));

        return Success(PrefixFound(containing_module, i));
    }

    /// Attempts to resolve the supplied name in the given module for the
    /// given namespace. If successful, returns the target corresponding to
    /// the name.
    ///
    /// The boolean returned on success is an indicator of whether this lookup
    /// passed through a public re-export proxy.
    fn resolve_name_in_module(&mut self,
                              module_: Rc<Module>,
                              name: Name,
                              namespace: Namespace,
                              name_search_type: NameSearchType,
                              allow_private_imports: bool)
                              -> ResolveResult<(Target, bool)> {
        debug!("(resolving name in module) resolving `{}` in `{}`",
               &token::get_name(name),
               self.module_to_string(&*module_));

        // First, check the direct children of the module.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        match module_.children.borrow().get(&name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                debug!("(resolving name in module) found node as child");
                return Success((Target::new(module_.clone(),
                                            name_bindings.clone(),
                                            Shadowable::Never),
                               false));
            }
            Some(_) | None => {
                // Continue.
            }
        }

        // Next, check the module's imports if necessary.

        // If this is a search of all imports, we should be done with glob
        // resolution at this point.
        if name_search_type == PathSearch {
            assert_eq!(module_.glob_count.get(), 0);
        }

        // Check the list of resolved imports.
        match module_.import_resolutions.borrow().get(&name) {
            Some(import_resolution) if allow_private_imports ||
                                       import_resolution.is_public => {

                if import_resolution.is_public &&
                        import_resolution.outstanding_references != 0 {
                    debug!("(resolving name in module) import \
                           unresolved; bailing out");
                    return Indeterminate;
                }
                match import_resolution.target_for_namespace(namespace) {
                    None => {
                        debug!("(resolving name in module) name found, \
                                but not in namespace {:?}",
                               namespace);
                    }
                    Some(target) => {
                        debug!("(resolving name in module) resolved to \
                                import");
                        // track used imports and extern crates as well
                        let id = import_resolution.id(namespace);
                        self.used_imports.insert((id, namespace));
                        self.record_import_use(id, name);
                        if let Some(DefId{krate: kid, ..}) = target.target_module.def_id.get() {
                            self.used_crates.insert(kid);
                        }
                        return Success((target, true));
                    }
                }
            }
            Some(..) | None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            // FIXME (21114): In principle unclear `child` *has* to be lifted.
            let child = module_.external_module_children.borrow().get(&name).cloned();
            if let Some(module) = child {
                let name_bindings =
                    Rc::new(Resolver::create_name_bindings_from_module(module));
                return Success((Target::new(module_,
                                            name_bindings,
                                            Shadowable::Never),
                                false));
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve `{}`",
               &token::get_name(name));
        return Failed(None);
    }

    fn report_unresolved_imports(&mut self, module_: Rc<Module>) {
        let index = module_.resolved_import_count.get();
        let imports = module_.imports.borrow();
        let import_count = imports.len();
        if index != import_count {
            let sn = self.session
                         .codemap()
                         .span_to_snippet((*imports)[index].span)
                         .unwrap();
            if sn.contains("::") {
                self.resolve_error((*imports)[index].span,
                                   "unresolved import");
            } else {
                let err = format!("unresolved import (maybe you meant `{}::*`?)",
                                  sn);
                self.resolve_error((*imports)[index].span, &err[..]);
            }
        }

        // Descend into children and anonymous children.
        build_reduced_graph::populate_module_if_necessary(self, &module_);

        for (_, child_node) in &*module_.children.borrow() {
            match child_node.get_module_if_available() {
                None => {
                    // Continue.
                }
                Some(child_module) => {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for (_, module_) in &*module_.anonymous_children.borrow() {
            self.report_unresolved_imports(module_.clone());
        }
    }

    // AST resolution
    //
    // We maintain a list of value ribs and type ribs.
    //
    // Simultaneously, we keep track of the current position in the module
    // graph in the `current_module` pointer. When we go to resolve a name in
    // the value or type namespaces, we first look through all the ribs and
    // then query the module graph. When we resolve a name in the module
    // namespace, we can skip all the ribs (since nested modules are not
    // allowed within blocks in Rust) and jump straight to the current module
    // graph node.
    //
    // Named implementations are handled separately. When we find a method
    // call, we consult the module node to find all of the implementations in
    // scope. This information is lazily cached in the module node. We then
    // generate a fake "implementation scope" containing all the
    // implementations thus found, for compatibility with old resolve pass.

    fn with_scope<F>(&mut self, name: Option<Name>, f: F) where
        F: FnOnce(&mut Resolver),
    {
        let orig_module = self.current_module.clone();

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                build_reduced_graph::populate_module_if_necessary(self, &orig_module);

                match orig_module.children.borrow().get(&name) {
                    None => {
                        debug!("!!! (with scope) didn't find `{}` in `{}`",
                               token::get_name(name),
                               self.module_to_string(&*orig_module));
                    }
                    Some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            None => {
                                debug!("!!! (with scope) didn't find module \
                                        for `{}` in `{}`",
                                       token::get_name(name),
                                       self.module_to_string(&*orig_module));
                            }
                            Some(module_) => {
                                self.current_module = module_;
                            }
                        }
                    }
                }
            }
        }

        f(self);

        self.current_module = orig_module;
    }

    /// Wraps the given definition in the appropriate number of `DefUpvar`
    /// wrappers.
    fn upvarify(&self,
                ribs: &[Rib],
                def_like: DefLike,
                span: Span)
                -> Option<DefLike> {
        match def_like {
            DlDef(d @ DefUpvar(..)) => {
                self.session.span_bug(span,
                    &format!("unexpected {:?} in bindings", d)[])
            }
            DlDef(d @ DefLocal(_)) => {
                let node_id = d.def_id().node;
                let mut def = d;
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind => {
                            // Nothing to do. Continue.
                        }
                        ClosureRibKind(function_id) => {
                            let prev_def = def;
                            def = DefUpvar(node_id, function_id);

                            let mut seen = self.freevars_seen.borrow_mut();
                            let seen = match seen.entry(function_id) {
                                Occupied(v) => v.into_mut(),
                                Vacant(v) => v.insert(NodeSet()),
                            };
                            if seen.contains(&node_id) {
                                continue;
                            }
                            match self.freevars.borrow_mut().entry(function_id) {
                                Occupied(v) => v.into_mut(),
                                Vacant(v) => v.insert(vec![]),
                            }.push(Freevar { def: prev_def, span: span });
                            seen.insert(node_id);
                        }
                        MethodRibKind(item_id, _) => {
                            // If the def is a ty param, and came from the parent
                            // item, it's ok
                            match def {
                                DefTyParam(_, _, did, _) if {
                                    self.def_map.borrow().get(&did.node).cloned()
                                        == Some(DefTyParamBinder(item_id))
                                } => {} // ok
                                DefSelfTy(did) if did == item_id => {} // ok
                                _ => {
                                    // This was an attempt to access an upvar inside a
                                    // named function item. This is not allowed, so we
                                    // report an error.

                                    self.resolve_error(
                                        span,
                                        "can't capture dynamic environment in a fn item; \
                                        use the || { ... } closure form instead");

                                    return None;
                                }
                            }
                        }
                        ItemRibKind => {
                            // This was an attempt to access an upvar inside a
                            // named function item. This is not allowed, so we
                            // report an error.

                            self.resolve_error(
                                span,
                                "can't capture dynamic environment in a fn item; \
                                use the || { ... } closure form instead");

                            return None;
                        }
                        ConstantItemRibKind => {
                            // Still doesn't deal with upvars
                            self.resolve_error(span,
                                               "attempt to use a non-constant \
                                                value in a constant");

                        }
                    }
                }
                Some(DlDef(def))
            }
            DlDef(def @ DefTyParam(..)) |
            DlDef(def @ DefSelfTy(..)) => {
                for rib in ribs {
                    match rib.kind {
                        NormalRibKind | ClosureRibKind(..) => {
                            // Nothing to do. Continue.
                        }
                        MethodRibKind(item_id, _) => {
                            // If the def is a ty param, and came from the parent
                            // item, it's ok
                            match def {
                                DefTyParam(_, _, did, _) if {
                                    self.def_map.borrow().get(&did.node).cloned()
                                        == Some(DefTyParamBinder(item_id))
                                } => {} // ok
                                DefSelfTy(did) if did == item_id => {} // ok

                                _ => {
                                    // This was an attempt to use a type parameter outside
                                    // its scope.

                                    self.resolve_error(span,
                                                        "can't use type parameters from \
                                                        outer function; try using a local \
                                                        type parameter instead");

                                    return None;
                                }
                            }
                        }
                        ItemRibKind => {
                            // This was an attempt to use a type parameter outside
                            // its scope.

                            self.resolve_error(span,
                                               "can't use type parameters from \
                                                outer function; try using a local \
                                                type parameter instead");

                            return None;
                        }
                        ConstantItemRibKind => {
                            // see #9186
                            self.resolve_error(span,
                                               "cannot use an outer type \
                                                parameter in this context");

                        }
                    }
                }
                Some(DlDef(def))
            }
            _ => Some(def_like)
        }
    }

    /// Searches the current set of local scopes and
    /// applies translations for closures.
    fn search_ribs(&self,
                   ribs: &[Rib],
                   name: Name,
                   span: Span)
                   -> Option<DefLike> {
        // FIXME #4950: Try caching?

        for (i, rib) in ribs.iter().enumerate().rev() {
            match rib.bindings.get(&name).cloned() {
                Some(def_like) => {
                    return self.upvarify(&ribs[i + 1..], def_like, span);
                }
                None => {
                    // Continue.
                }
            }
        }

        None
    }

    /// Searches the current set of local scopes for labels.
    /// Stops after meeting a closure.
    fn search_label(&self, name: Name) -> Option<DefLike> {
        for rib in self.label_ribs.iter().rev() {
            match rib.kind {
                NormalRibKind => {
                    // Continue
                }
                _ => {
                    // Do not resolve labels across function boundary
                    return None
                }
            }
            let result = rib.bindings.get(&name).cloned();
            if result.is_some() {
                return result
            }
        }
        None
    }

    fn resolve_crate(&mut self, krate: &ast::Crate) {
        debug!("(resolving crate) starting");

        visit::walk_crate(self, krate);
    }

    fn check_if_primitive_type_name(&self, name: Name, span: Span) {
        if let Some(_) = self.primitive_type_table.primitive_types.get(&name) {
            span_err!(self.session, span, E0317,
                "user-defined types or type parameters cannot shadow the primitive types");
        }
    }

    fn resolve_item(&mut self, item: &Item) {
        let name = item.ident.name;

        debug!("(resolving item) resolving {}",
               token::get_name(name));

        match item.node {

            // enum item: resolve all the variants' discrs,
            // then resolve the ty params
            ItemEnum(ref enum_def, ref generics) => {
                self.check_if_primitive_type_name(name, item.span);

                for variant in &(*enum_def).variants {
                    if let Some(ref dis_expr) = variant.node.disr_expr {
                        // resolve the discriminator expr
                        // as a constant
                        self.with_constant_rib(|this| {
                            this.resolve_expr(&**dis_expr);
                        });
                    }
                }

                // n.b. the discr expr gets visited twice.
                // but maybe it's okay since the first time will signal an
                // error if there is one? -- tjc
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               TypeSpace,
                                                               item.id,
                                                               ItemRibKind),
                                             |this| {
                    this.resolve_type_parameters(&generics.ty_params);
                    this.resolve_where_clause(&generics.where_clause);
                    visit::walk_item(this, item);
                });
            }

            ItemTy(_, ref generics) => {
                self.check_if_primitive_type_name(name, item.span);

                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               TypeSpace,
                                                               item.id,
                                                               ItemRibKind),
                                             |this| {
                    this.resolve_type_parameters(&generics.ty_params);
                    visit::walk_item(this, item);
                });
            }

            ItemImpl(_, _,
                     ref generics,
                     ref implemented_traits,
                     ref self_type,
                     ref impl_items) => {
                self.resolve_implementation(item.id,
                                            generics,
                                            implemented_traits,
                                            &**self_type,
                                            &impl_items[..]);
            }

            ItemTrait(_, ref generics, ref bounds, ref trait_items) => {
                self.check_if_primitive_type_name(name, item.span);

                // Create a new rib for the self type.
                let mut self_type_rib = Rib::new(ItemRibKind);

                // plain insert (no renaming, types are not currently hygienic....)
                let name = self.type_self_name;
                self_type_rib.bindings.insert(name, DlDef(DefSelfTy(item.id)));
                self.type_ribs.push(self_type_rib);

                // Create a new rib for the trait-wide type parameters.
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               TypeSpace,
                                                               item.id,
                                                               NormalRibKind),
                                             |this| {
                    this.resolve_type_parameters(&generics.ty_params);
                    this.resolve_where_clause(&generics.where_clause);

                    this.resolve_type_parameter_bounds(item.id, bounds,
                                                       TraitDerivation);

                    for trait_item in &(*trait_items) {
                        // Create a new rib for the trait_item-specific type
                        // parameters.
                        //
                        // FIXME #4951: Do we need a node ID here?

                        match *trait_item {
                          ast::RequiredMethod(ref ty_m) => {
                            this.with_type_parameter_rib
                                (HasTypeParameters(&ty_m.generics,
                                                   FnSpace,
                                                   item.id,
                                        MethodRibKind(item.id, RequiredMethod)),
                                 |this| {

                                // Resolve the method-specific type
                                // parameters.
                                this.resolve_type_parameters(
                                    &ty_m.generics.ty_params);
                                this.resolve_where_clause(&ty_m.generics
                                                               .where_clause);

                                for argument in &ty_m.decl.inputs {
                                    this.resolve_type(&*argument.ty);
                                }

                                if let SelfExplicit(ref typ, _) = ty_m.explicit_self.node {
                                    this.resolve_type(&**typ)
                                }

                                if let ast::Return(ref ret_ty) = ty_m.decl.output {
                                    this.resolve_type(&**ret_ty);
                                }
                            });
                          }
                          ast::ProvidedMethod(ref m) => {
                              this.resolve_method(MethodRibKind(item.id,
                                                                ProvidedMethod(m.id)),
                                                  &**m)
                          }
                          ast::TypeTraitItem(ref data) => {
                              this.resolve_type_parameter(&data.ty_param);
                              visit::walk_trait_item(this, trait_item);
                          }
                        }
                    }
                });

                self.type_ribs.pop();
            }

            ItemStruct(ref struct_def, ref generics) => {
                self.check_if_primitive_type_name(name, item.span);

                self.resolve_struct(item.id,
                                    generics,
                                    &struct_def.fields[]);
            }

            ItemMod(ref module_) => {
                self.with_scope(Some(name), |this| {
                    this.resolve_module(module_, item.span, name,
                                        item.id);
                });
            }

            ItemForeignMod(ref foreign_module) => {
                self.with_scope(Some(name), |this| {
                    for foreign_item in &foreign_module.items {
                        match foreign_item.node {
                            ForeignItemFn(_, ref generics) => {
                                this.with_type_parameter_rib(
                                    HasTypeParameters(
                                        generics, FnSpace, foreign_item.id,
                                        ItemRibKind),
                                    |this| {
                                        this.resolve_type_parameters(&generics.ty_params);
                                        this.resolve_where_clause(&generics.where_clause);
                                        visit::walk_foreign_item(this, &**foreign_item)
                                    });
                            }
                            ForeignItemStatic(..) => {
                                visit::walk_foreign_item(this,
                                                         &**foreign_item);
                            }
                        }
                    }
                });
            }

            ItemFn(ref fn_decl, _, _, ref generics, ref block) => {
                self.resolve_function(ItemRibKind,
                                      Some(&**fn_decl),
                                      HasTypeParameters
                                        (generics,
                                         FnSpace,
                                         item.id,
                                         ItemRibKind),
                                      &**block);
            }

            ItemConst(..) | ItemStatic(..) => {
                self.with_constant_rib(|this| {
                    visit::walk_item(this, item);
                });
            }

            ItemUse(ref view_path) => {
                // check for imports shadowing primitive types
                if let ast::ViewPathSimple(ident, _) = view_path.node {
                    match self.def_map.borrow().get(&item.id) {
                        Some(&DefTy(..)) | Some(&DefStruct(..)) | Some(&DefTrait(..)) | None => {
                            self.check_if_primitive_type_name(ident.name, item.span);
                        }
                        _ => {}
                    }
                }
            }

            ItemExternCrate(_) | ItemMac(..) => {
                // do nothing, these are just around to be encoded
            }
        }
    }

    fn with_type_parameter_rib<F>(&mut self, type_parameters: TypeParameters, f: F) where
        F: FnOnce(&mut Resolver),
    {
        match type_parameters {
            HasTypeParameters(generics, space, node_id, rib_kind) => {
                let mut function_type_rib = Rib::new(rib_kind);
                let mut seen_bindings = HashSet::new();
                for (index, type_parameter) in generics.ty_params.iter().enumerate() {
                    let name = type_parameter.ident.name;
                    debug!("with_type_parameter_rib: {} {}", node_id,
                           type_parameter.id);

                    if seen_bindings.contains(&name) {
                        self.resolve_error(type_parameter.span,
                                           &format!("the name `{}` is already \
                                                    used for a type \
                                                    parameter in this type \
                                                    parameter list",
                                                   token::get_name(
                                                       name))[])
                    }
                    seen_bindings.insert(name);

                    let def_like = DlDef(DefTyParam(space,
                                                    index as u32,
                                                    local_def(type_parameter.id),
                                                    name));
                    // Associate this type parameter with
                    // the item that bound it
                    self.record_def(type_parameter.id,
                                    (DefTyParamBinder(node_id), LastMod(AllPublic)));
                    // plain insert (no renaming)
                    function_type_rib.bindings.insert(name, def_like);
                }
                self.type_ribs.push(function_type_rib);
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        match type_parameters {
            HasTypeParameters(..) => { self.type_ribs.pop(); }
            NoTypeParameters => { }
        }
    }

    fn with_label_rib<F>(&mut self, f: F) where
        F: FnOnce(&mut Resolver),
    {
        self.label_ribs.push(Rib::new(NormalRibKind));
        f(self);
        self.label_ribs.pop();
    }

    fn with_constant_rib<F>(&mut self, f: F) where
        F: FnOnce(&mut Resolver),
    {
        self.value_ribs.push(Rib::new(ConstantItemRibKind));
        self.type_ribs.push(Rib::new(ConstantItemRibKind));
        f(self);
        self.type_ribs.pop();
        self.value_ribs.pop();
    }

    fn resolve_function(&mut self,
                        rib_kind: RibKind,
                        optional_declaration: Option<&FnDecl>,
                        type_parameters: TypeParameters,
                        block: &Block) {
        // Create a value rib for the function.
        let function_value_rib = Rib::new(rib_kind);
        self.value_ribs.push(function_value_rib);

        // Create a label rib for the function.
        let function_label_rib = Rib::new(rib_kind);
        self.label_ribs.push(function_label_rib);

        // If this function has type parameters, add them now.
        self.with_type_parameter_rib(type_parameters, |this| {
            // Resolve the type parameters.
            match type_parameters {
                NoTypeParameters => {
                    // Continue.
                }
                HasTypeParameters(ref generics, _, _, _) => {
                    this.resolve_type_parameters(&generics.ty_params);
                    this.resolve_where_clause(&generics.where_clause);
                }
            }

            // Add each argument to the rib.
            match optional_declaration {
                None => {
                    // Nothing to do.
                }
                Some(declaration) => {
                    let mut bindings_list = HashMap::new();
                    for argument in &declaration.inputs {
                        this.resolve_pattern(&*argument.pat,
                                             ArgumentIrrefutableMode,
                                             &mut bindings_list);

                        this.resolve_type(&*argument.ty);

                        debug!("(resolving function) recorded argument");
                    }

                    if let ast::Return(ref ret_ty) = declaration.output {
                        this.resolve_type(&**ret_ty);
                    }
                }
            }

            // Resolve the function body.
            this.resolve_block(&*block);

            debug!("(resolving function) leaving function");
        });

        self.label_ribs.pop();
        self.value_ribs.pop();
    }

    fn resolve_type_parameters(&mut self,
                               type_parameters: &OwnedSlice<TyParam>) {
        for type_parameter in &**type_parameters {
            self.resolve_type_parameter(type_parameter);
        }
    }

    fn resolve_type_parameter(&mut self,
                              type_parameter: &TyParam) {
        self.check_if_primitive_type_name(type_parameter.ident.name, type_parameter.span);
        for bound in &*type_parameter.bounds {
            self.resolve_type_parameter_bound(type_parameter.id, bound,
                                              TraitBoundingTypeParameter);
        }
        match type_parameter.default {
            Some(ref ty) => self.resolve_type(&**ty),
            None => {}
        }
    }

    fn resolve_type_parameter_bounds(&mut self,
                                     id: NodeId,
                                     type_parameter_bounds: &OwnedSlice<TyParamBound>,
                                     reference_type: TraitReferenceType) {
        for type_parameter_bound in &**type_parameter_bounds {
            self.resolve_type_parameter_bound(id, type_parameter_bound,
                                              reference_type);
        }
    }

    fn resolve_type_parameter_bound(&mut self,
                                    id: NodeId,
                                    type_parameter_bound: &TyParamBound,
                                    reference_type: TraitReferenceType) {
        match *type_parameter_bound {
            TraitTyParamBound(ref tref, _) => {
                self.resolve_poly_trait_reference(id, tref, reference_type)
            }
            RegionTyParamBound(..) => {}
        }
    }

    fn resolve_poly_trait_reference(&mut self,
                                    id: NodeId,
                                    poly_trait_reference: &PolyTraitRef,
                                    reference_type: TraitReferenceType) {
        self.resolve_trait_reference(id, &poly_trait_reference.trait_ref, reference_type)
    }

    fn resolve_trait_reference(&mut self,
                               id: NodeId,
                               trait_reference: &TraitRef,
                               reference_type: TraitReferenceType) {
        match self.resolve_path(id, &trait_reference.path, TypeNS, true) {
            None => {
                let path_str = self.path_names_to_string(&trait_reference.path);
                let usage_str = match reference_type {
                    TraitBoundingTypeParameter => "bound type parameter with",
                    TraitImplementation        => "implement",
                    TraitDerivation            => "derive",
                    TraitObject                => "reference",
                    TraitQPath                 => "extract an associated item from",
                };

                let msg = format!("attempt to {} a nonexistent trait `{}`", usage_str, path_str);
                self.resolve_error(trait_reference.path.span, &msg[..]);
            }
            Some(def) => {
                match def {
                    (DefTrait(_), _) => {
                        debug!("(resolving trait) found trait def: {:?}", def);
                        self.record_def(trait_reference.ref_id, def);
                    }
                    (def, _) => {
                        self.resolve_error(trait_reference.path.span,
                                           &format!("`{}` is not a trait",
                                                   self.path_names_to_string(
                                                       &trait_reference.path))[]);

                        // If it's a typedef, give a note
                        if let DefTy(..) = def {
                            self.session.span_note(
                                trait_reference.path.span,
                                &format!("`type` aliases cannot be used for traits")
                                []);
                        }
                    }
                }
            }
        }
    }

    fn resolve_where_clause(&mut self, where_clause: &ast::WhereClause) {
        for predicate in &where_clause.predicates {
            match predicate {
                &ast::WherePredicate::BoundPredicate(ref bound_pred) => {
                    self.resolve_type(&*bound_pred.bounded_ty);

                    for bound in &*bound_pred.bounds {
                        self.resolve_type_parameter_bound(bound_pred.bounded_ty.id, bound,
                                                          TraitBoundingTypeParameter);
                    }
                }
                &ast::WherePredicate::RegionPredicate(_) => {}
                &ast::WherePredicate::EqPredicate(ref eq_pred) => {
                    match self.resolve_path(eq_pred.id, &eq_pred.path, TypeNS, true) {
                        Some((def @ DefTyParam(..), last_private)) => {
                            self.record_def(eq_pred.id, (def, last_private));
                        }
                        _ => {
                            self.resolve_error(eq_pred.path.span,
                                               "undeclared associated type");
                        }
                    }

                    self.resolve_type(&*eq_pred.ty);
                }
            }
        }
    }

    fn resolve_struct(&mut self,
                      id: NodeId,
                      generics: &Generics,
                      fields: &[StructField]) {
        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       TypeSpace,
                                                       id,
                                                       ItemRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.resolve_type_parameters(&generics.ty_params);
            this.resolve_where_clause(&generics.where_clause);

            // Resolve fields.
            for field in fields {
                this.resolve_type(&*field.node.ty);
            }
        });
    }

    // Does this really need to take a RibKind or is it always going
    // to be NormalRibKind?
    fn resolve_method(&mut self,
                      rib_kind: RibKind,
                      method: &ast::Method) {
        let method_generics = method.pe_generics();
        let type_parameters = HasTypeParameters(method_generics,
                                                FnSpace,
                                                method.id,
                                                rib_kind);

        if let SelfExplicit(ref typ, _) = method.pe_explicit_self().node {
            self.resolve_type(&**typ);
        }

        self.resolve_function(rib_kind,
                              Some(method.pe_fn_decl()),
                              type_parameters,
                              method.pe_body());
    }

    fn with_current_self_type<T, F>(&mut self, self_type: &Ty, f: F) -> T where
        F: FnOnce(&mut Resolver) -> T,
    {
        // Handle nested impls (inside fn bodies)
        let previous_value = replace(&mut self.current_self_type, Some(self_type.clone()));
        let result = f(self);
        self.current_self_type = previous_value;
        result
    }

    fn with_optional_trait_ref<T, F>(&mut self, id: NodeId,
                                     opt_trait_ref: &Option<TraitRef>,
                                     f: F) -> T where
        F: FnOnce(&mut Resolver) -> T,
    {
        let new_val = match *opt_trait_ref {
            Some(ref trait_ref) => {
                self.resolve_trait_reference(id, trait_ref, TraitImplementation);

                match self.def_map.borrow().get(&trait_ref.ref_id) {
                    Some(def) => {
                        let did = def.def_id();
                        Some((did, trait_ref.clone()))
                    }
                    None => None
                }
            }
            None => None
        };
        let original_trait_ref = replace(&mut self.current_trait_ref, new_val);
        let result = f(self);
        self.current_trait_ref = original_trait_ref;
        result
    }

    fn resolve_implementation(&mut self,
                              id: NodeId,
                              generics: &Generics,
                              opt_trait_reference: &Option<TraitRef>,
                              self_type: &Ty,
                              impl_items: &[ImplItem]) {
        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       TypeSpace,
                                                       id,
                                                       NormalRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.resolve_type_parameters(&generics.ty_params);
            this.resolve_where_clause(&generics.where_clause);

            // Resolve the trait reference, if necessary.
            this.with_optional_trait_ref(id, opt_trait_reference, |this| {
                // Resolve the self type.
                this.resolve_type(self_type);

                this.with_current_self_type(self_type, |this| {
                    for impl_item in impl_items {
                        match *impl_item {
                            MethodImplItem(ref method) => {
                                // If this is a trait impl, ensure the method
                                // exists in trait
                                this.check_trait_item(method.pe_ident().name,
                                                      method.span);

                                // We also need a new scope for the method-
                                // specific type parameters.
                                this.resolve_method(
                                    MethodRibKind(id, ProvidedMethod(method.id)),
                                    &**method);
                            }
                            TypeImplItem(ref typedef) => {
                                // If this is a trait impl, ensure the method
                                // exists in trait
                                this.check_trait_item(typedef.ident.name,
                                                      typedef.span);

                                this.resolve_type(&*typedef.typ);
                            }
                        }
                    }
                });
            });
        });

        // Check that the current type is indeed a type, if we have an anonymous impl
        if opt_trait_reference.is_none() {
            match self_type.node {
                // TyPath is the only thing that we handled in `build_reduced_graph_for_item`,
                // where we created a module with the name of the type in order to implement
                // an anonymous trait. In the case that the path does not resolve to an actual
                // type, the result will be that the type name resolves to a module but not
                // a type (shadowing any imported modules or types with this name), leading
                // to weird user-visible bugs. So we ward this off here. See #15060.
                TyPath(ref path, path_id) => {
                    match self.def_map.borrow().get(&path_id) {
                        // FIXME: should we catch other options and give more precise errors?
                        Some(&DefMod(_)) => {
                            self.resolve_error(path.span, "inherent implementations are not \
                                                           allowed for types not defined in \
                                                           the current module");
                        }
                        _ => {}
                    }
                }
                _ => { }
            }
        }
    }

    fn check_trait_item(&self, name: Name, span: Span) {
        // If there is a TraitRef in scope for an impl, then the method must be in the trait.
        if let Some((did, ref trait_ref)) = self.current_trait_ref {
            if self.trait_item_map.get(&(name, did)).is_none() {
                let path_str = self.path_names_to_string(&trait_ref.path);
                self.resolve_error(span,
                                    &format!("method `{}` is not a member of trait `{}`",
                                            token::get_name(name),
                                            path_str)[]);
            }
        }
    }

    fn resolve_module(&mut self, module: &Mod, _span: Span,
                      _name: Name, id: NodeId) {
        // Write the implementations in scope into the module metadata.
        debug!("(resolving module) resolving module ID {}", id);
        visit::walk_mod(self, module);
    }

    fn resolve_local(&mut self, local: &Local) {
        // Resolve the type.
        if let Some(ref ty) = local.ty {
            self.resolve_type(&**ty);
        }

        // Resolve the initializer, if necessary.
        match local.init {
            None => {
                // Nothing to do.
            }
            Some(ref initializer) => {
                self.resolve_expr(&**initializer);
            }
        }

        // Resolve the pattern.
        let mut bindings_list = HashMap::new();
        self.resolve_pattern(&*local.pat,
                             LocalIrrefutableMode,
                             &mut bindings_list);
    }

    // build a map from pattern identifiers to binding-info's.
    // this is done hygienically. This could arise for a macro
    // that expands into an or-pattern where one 'x' was from the
    // user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut result = HashMap::new();
        pat_bindings(&self.def_map, pat, |binding_mode, _id, sp, path1| {
            let name = mtwt::resolve(path1.node);
            result.insert(name, BindingInfo {
                span: sp,
                binding_mode: binding_mode
            });
        });
        return result;
    }

    // check that all of the arms in an or-pattern have exactly the
    // same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, arm: &Arm) {
        if arm.pats.len() == 0 {
            return
        }
        let map_0 = self.binding_mode_map(&*arm.pats[0]);
        for (i, p) in arm.pats.iter().enumerate() {
            let map_i = self.binding_mode_map(&**p);

            for (&key, &binding_0) in &map_0 {
                match map_i.get(&key) {
                  None => {
                    self.resolve_error(
                        p.span,
                        &format!("variable `{}` from pattern #1 is \
                                  not bound in pattern #{}",
                                token::get_name(key),
                                i + 1)[]);
                  }
                  Some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.resolve_error(
                            binding_i.span,
                            &format!("variable `{}` is bound with different \
                                      mode in pattern #{} than in pattern #1",
                                    token::get_name(key),
                                    i + 1)[]);
                    }
                  }
                }
            }

            for (&key, &binding) in &map_i {
                if !map_0.contains_key(&key) {
                    self.resolve_error(
                        binding.span,
                        &format!("variable `{}` from pattern {}{} is \
                                  not bound in pattern {}1",
                                token::get_name(key),
                                "#", i + 1, "#")[]);
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        self.value_ribs.push(Rib::new(NormalRibKind));

        let mut bindings_list = HashMap::new();
        for pattern in &arm.pats {
            self.resolve_pattern(&**pattern, RefutableMode, &mut bindings_list);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        visit::walk_expr_opt(self, &arm.guard);
        self.resolve_expr(&*arm.body);

        self.value_ribs.pop();
    }

    fn resolve_block(&mut self, block: &Block) {
        debug!("(resolving block) entering block");
        self.value_ribs.push(Rib::new(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module.clone();
        match orig_module.anonymous_children.borrow().get(&block.id) {
            None => { /* Nothing to do. */ }
            Some(anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module.clone();
            }
        }

        // Check for imports appearing after non-item statements.
        let mut found_non_item = false;
        for statement in &block.stmts {
            if let ast::StmtDecl(ref declaration, _) = statement.node {
                if let ast::DeclItem(ref i) = declaration.node {
                    match i.node {
                        ItemExternCrate(_) | ItemUse(_) if found_non_item => {
                            span_err!(self.session, i.span, E0154,
                                "imports are not allowed after non-item statements");
                        }
                        _ => {}
                    }
                } else {
                    found_non_item = true
                }
            } else {
                found_non_item = true;
            }
        }

        // Descend into the block.
        visit::walk_block(self, block);

        // Move back up.
        self.current_module = orig_module;

        self.value_ribs.pop();
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            TyPath(ref path, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // looking for it.
                let mut result_def = None;

                // First, check to see whether the name is a primitive type.
                if path.segments.len() == 1 {
                    let id = path.segments.last().unwrap().identifier;

                    match self.primitive_type_table
                            .primitive_types
                            .get(&id.name) {

                        Some(&primitive_type) => {
                            result_def =
                                Some((DefPrimTy(primitive_type), LastMod(AllPublic)));

                            if path.segments[0].parameters.has_lifetimes() {
                                span_err!(self.session, path.span, E0157,
                                    "lifetime parameters are not allowed on this type");
                            } else if !path.segments[0].parameters.is_empty() {
                                span_err!(self.session, path.span, E0153,
                                    "type parameters are not allowed on this type");
                            }
                        }
                        None => {
                            // Continue.
                        }
                    }
                }

                if let None = result_def {
                    result_def = self.resolve_path(ty.id, path, TypeNS, true);
                }

                match result_def {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving type) writing resolution for `{}` \
                                (id {}) = {:?}",
                               self.path_names_to_string(path),
                               path_id, def);
                        self.record_def(path_id, def);
                    }
                    None => {
                        let msg = format!("use of undeclared type name `{}`",
                                          self.path_names_to_string(path));
                        self.resolve_error(ty.span, &msg[..]);
                    }
                }
            }

            TyObjectSum(ref ty, ref bound_vec) => {
                self.resolve_type(&**ty);
                self.resolve_type_parameter_bounds(ty.id, bound_vec,
                                                       TraitBoundingTypeParameter);
            }

            TyQPath(ref qpath) => {
                self.resolve_type(&*qpath.self_type);
                self.resolve_trait_reference(ty.id, &*qpath.trait_ref, TraitQPath);
                for ty in qpath.item_path.parameters.types() {
                    self.resolve_type(&**ty);
                }
                for binding in qpath.item_path.parameters.bindings() {
                    self.resolve_type(&*binding.ty);
                }
            }

            TyPolyTraitRef(ref bounds) => {
                self.resolve_type_parameter_bounds(
                    ty.id,
                    bounds,
                    TraitObject);
                visit::walk_ty(self, ty);
            }
            _ => {
                // Just resolve embedded types.
                visit::walk_ty(self, ty);
            }
        }
    }

    fn resolve_pattern(&mut self,
                       pattern: &Pat,
                       mode: PatternBindingMode,
                       // Maps idents to the node ID for the (outermost)
                       // pattern that binds them
                       bindings_list: &mut HashMap<Name, NodeId>) {
        let pat_id = pattern.id;
        walk_pat(pattern, |pattern| {
            match pattern.node {
                PatIdent(binding_mode, ref path1, _) => {

                    // The meaning of pat_ident with no type parameters
                    // depends on whether an enum variant or unit-like struct
                    // with that name is in scope. The probing lookup has to
                    // be careful not to emit spurious errors. Only matching
                    // patterns (match) can match nullary variants or
                    // unit-like structs. For binding patterns (let), matching
                    // such a value is simply disallowed (since it's rarely
                    // what you want).

                    let ident = path1.node;
                    let renamed = mtwt::resolve(ident);

                    match self.resolve_bare_identifier_pattern(ident.name, pattern.span) {
                        FoundStructOrEnumVariant(ref def, lp)
                                if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    struct or enum variant",
                                   token::get_name(renamed));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "an enum variant");
                            self.record_def(pattern.id, (def.clone(), lp));
                        }
                        FoundStructOrEnumVariant(..) => {
                            self.resolve_error(
                                pattern.span,
                                &format!("declaration of `{}` shadows an enum \
                                         variant or unit-like struct in \
                                         scope",
                                        token::get_name(renamed))[]);
                        }
                        FoundConst(ref def, lp) if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    constant",
                                   token::get_name(renamed));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "a constant");
                            self.record_def(pattern.id, (def.clone(), lp));
                        }
                        FoundConst(..) => {
                            self.resolve_error(pattern.span,
                                                  "only irrefutable patterns \
                                                   allowed here");
                        }
                        BareIdentifierPatternUnresolved => {
                            debug!("(resolving pattern) binding `{}`",
                                   token::get_name(renamed));

                            let def = DefLocal(pattern.id);

                            // Record the definition so that later passes
                            // will be able to distinguish variants from
                            // locals in patterns.

                            self.record_def(pattern.id, (def, LastMod(AllPublic)));

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)
                            if !bindings_list.contains_key(&renamed) {
                                let this = &mut *self;
                                let last_rib = this.value_ribs.last_mut().unwrap();
                                last_rib.bindings.insert(renamed, DlDef(def));
                                bindings_list.insert(renamed, pat_id);
                            } else if mode == ArgumentIrrefutableMode &&
                                    bindings_list.contains_key(&renamed) {
                                // Forbid duplicate bindings in the same
                                // parameter list.
                                self.resolve_error(pattern.span,
                                                   &format!("identifier `{}` \
                                                            is bound more \
                                                            than once in \
                                                            this parameter \
                                                            list",
                                                           token::get_ident(
                                                               ident))
                                                   [])
                            } else if bindings_list.get(&renamed) ==
                                    Some(&pat_id) {
                                // Then this is a duplicate variable in the
                                // same disjunction, which is an error.
                                self.resolve_error(pattern.span,
                                    &format!("identifier `{}` is bound \
                                             more than once in the same \
                                             pattern",
                                            token::get_ident(ident))[]);
                            }
                            // Else, not bound in the same pattern: do
                            // nothing.
                        }
                    }
                }

                PatEnum(ref path, _) => {
                    // This must be an enum variant, struct or const.
                    match self.resolve_path(pat_id, path, ValueNS, false) {
                        Some(def @ (DefVariant(..), _)) |
                        Some(def @ (DefStruct(..), _))  |
                        Some(def @ (DefConst(..), _)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some((DefStatic(..), _)) => {
                            self.resolve_error(path.span,
                                               "static variables cannot be \
                                                referenced in a pattern, \
                                                use a `const` instead");
                        }
                        Some(_) => {
                            self.resolve_error(path.span,
                                &format!("`{}` is not an enum variant, struct or const",
                                    token::get_ident(
                                        path.segments.last().unwrap().identifier)));
                        }
                        None => {
                            self.resolve_error(path.span,
                                &format!("unresolved enum variant, struct or const `{}`",
                                    token::get_ident(path.segments.last().unwrap().identifier)));
                        }
                    }

                    // Check the types in the path pattern.
                    for ty in path.segments
                                  .iter()
                                  .flat_map(|s| s.parameters.types().into_iter()) {
                        self.resolve_type(&**ty);
                    }
                }

                PatLit(ref expr) => {
                    self.resolve_expr(&**expr);
                }

                PatRange(ref first_expr, ref last_expr) => {
                    self.resolve_expr(&**first_expr);
                    self.resolve_expr(&**last_expr);
                }

                PatStruct(ref path, _, _) => {
                    match self.resolve_path(pat_id, path, TypeNS, false) {
                        Some(definition) => {
                            self.record_def(pattern.id, definition);
                        }
                        result => {
                            debug!("(resolving pattern) didn't find struct \
                                    def: {:?}", result);
                            let msg = format!("`{}` does not name a structure",
                                              self.path_names_to_string(path));
                            self.resolve_error(path.span, &msg[..]);
                        }
                    }
                }

                _ => {
                    // Nothing to do.
                }
            }
            true
        });
    }

    fn resolve_bare_identifier_pattern(&mut self, name: Name, span: Span)
                                       -> BareIdentifierPatternResolution {
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module,
                                                 name,
                                                 ValueNS) {
            Success((target, _)) => {
                debug!("(resolve bare identifier pattern) succeeded in \
                         finding {} at {:?}",
                        token::get_name(name),
                        target.bindings.value_def.borrow());
                match *target.bindings.value_def.borrow() {
                    None => {
                        panic!("resolved name in the value namespace to a \
                              set of name bindings with no def?!");
                    }
                    Some(def) => {
                        // For the two success cases, this lookup can be
                        // considered as not having a private component because
                        // the lookup happened only within the current module.
                        match def.def {
                            def @ DefVariant(..) | def @ DefStruct(..) => {
                                return FoundStructOrEnumVariant(def, LastMod(AllPublic));
                            }
                            def @ DefConst(..) => {
                                return FoundConst(def, LastMod(AllPublic));
                            }
                            DefStatic(..) => {
                                self.resolve_error(span,
                                                   "static variables cannot be \
                                                    referenced in a pattern, \
                                                    use a `const` instead");
                                return BareIdentifierPatternUnresolved;
                            }
                            _ => {
                                return BareIdentifierPatternUnresolved;
                            }
                        }
                    }
                }
            }

            Indeterminate => {
                panic!("unexpected indeterminate result");
            }
            Failed(err) => {
                match err {
                    Some((span, msg)) => {
                        self.resolve_error(span, &format!("failed to resolve: {}",
                                                         msg)[]);
                    }
                    None => ()
                }

                debug!("(resolve bare identifier pattern) failed to find {}",
                        token::get_name(name));
                return BareIdentifierPatternUnresolved;
            }
        }
    }

    /// If `check_ribs` is true, checks the local definitions first; i.e.
    /// doesn't skip straight to the containing module.
    fn resolve_path(&mut self,
                    id: NodeId,
                    path: &Path,
                    namespace: Namespace,
                    check_ribs: bool) -> Option<(Def, LastPrivate)> {
        // First, resolve the types and associated type bindings.
        for ty in path.segments.iter().flat_map(|s| s.parameters.types().into_iter()) {
            self.resolve_type(&**ty);
        }
        for binding in path.segments.iter().flat_map(|s| s.parameters.bindings().into_iter()) {
            self.resolve_type(&*binding.ty);
        }

        // A special case for sugared associated type paths `T::A` where `T` is
        // a type parameter and `A` is an associated type on some bound of `T`.
        if namespace == TypeNS && path.segments.len() == 2 {
            match self.resolve_identifier(path.segments[0].identifier,
                                          TypeNS,
                                          true,
                                          path.span) {
                Some((def, last_private)) => {
                    match def {
                        DefTyParam(_, _, did, _) => {
                            let def = DefAssociatedPath(TyParamProvenance::FromParam(did),
                                                        path.segments.last()
                                                            .unwrap().identifier);
                            return Some((def, last_private));
                        }
                        DefSelfTy(nid) => {
                            let def = DefAssociatedPath(TyParamProvenance::FromSelf(local_def(nid)),
                                                        path.segments.last()
                                                            .unwrap().identifier);
                            return Some((def, last_private));
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        if path.global {
            return self.resolve_crate_relative_path(path, namespace);
        }

        // Try to find a path to an item in a module.
        let unqualified_def =
                self.resolve_identifier(path.segments.last().unwrap().identifier,
                                        namespace,
                                        check_ribs,
                                        path.span);

        if path.segments.len() > 1 {
            let def = self.resolve_module_relative_path(path, namespace);
            match (def, unqualified_def) {
                (Some((ref d, _)), Some((ref ud, _))) if *d == *ud => {
                    self.session
                        .add_lint(lint::builtin::UNUSED_QUALIFICATIONS,
                                  id,
                                  path.span,
                                  "unnecessary qualification".to_string());
                }
                _ => ()
            }

            return def;
        }

        return unqualified_def;
    }

    // resolve a single identifier (used as a varref)
    fn resolve_identifier(&mut self,
                          identifier: Ident,
                          namespace: Namespace,
                          check_ribs: bool,
                          span: Span)
                          -> Option<(Def, LastPrivate)> {
        if check_ribs {
            match self.resolve_identifier_in_local_ribs(identifier,
                                                        namespace,
                                                        span) {
                Some(def) => {
                    return Some((def, LastMod(AllPublic)));
                }
                None => {
                    // Continue.
                }
            }
        }

        return self.resolve_item_by_name_in_lexical_scope(identifier.name, namespace);
    }

    // FIXME #4952: Merge me with resolve_name_in_module?
    fn resolve_definition_of_name_in_module(&mut self,
                                            containing_module: Rc<Module>,
                                            name: Name,
                                            namespace: Namespace)
                                            -> NameDefinition {
        // First, search children.
        build_reduced_graph::populate_module_if_necessary(self, &containing_module);

        match containing_module.children.borrow().get(&name) {
            Some(child_name_bindings) => {
                match child_name_bindings.def_for_namespace(namespace) {
                    Some(def) => {
                        // Found it. Stop the search here.
                        let p = child_name_bindings.defined_in_public_namespace(
                                        namespace);
                        let lp = if p {LastMod(AllPublic)} else {
                            LastMod(DependsOn(def.def_id()))
                        };
                        return ChildNameDefinition(def, lp);
                    }
                    None => {}
                }
            }
            None => {}
        }

        // Next, search import resolutions.
        match containing_module.import_resolutions.borrow().get(&name) {
            Some(import_resolution) if import_resolution.is_public => {
                if let Some(target) = (*import_resolution).target_for_namespace(namespace) {
                    match target.bindings.def_for_namespace(namespace) {
                        Some(def) => {
                            // Found it.
                            let id = import_resolution.id(namespace);
                            // track imports and extern crates as well
                            self.used_imports.insert((id, namespace));
                            self.record_import_use(id, name);
                            match target.target_module.def_id.get() {
                                Some(DefId{krate: kid, ..}) => {
                                    self.used_crates.insert(kid);
                                },
                                _ => {}
                            }
                            return ImportNameDefinition(def, LastMod(AllPublic));
                        }
                        None => {
                            // This can happen with external impls, due to
                            // the imperfect way we read the metadata.
                        }
                    }
                }
            }
            Some(..) | None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            if let Some(module) = containing_module.external_module_children.borrow()
                                                   .get(&name).cloned() {
                if let Some(def_id) = module.def_id.get() {
                    // track used crates
                    self.used_crates.insert(def_id.krate);
                    let lp = if module.is_public {LastMod(AllPublic)} else {
                        LastMod(DependsOn(def_id))
                    };
                    return ChildNameDefinition(DefMod(def_id), lp);
                }
            }
        }

        return NoNameDefinition;
    }

    // resolve a "module-relative" path, e.g. a::b::c
    fn resolve_module_relative_path(&mut self,
                                    path: &Path,
                                    namespace: Namespace)
                                    -> Option<(Def, LastPrivate)> {
        let module_path = path.segments.init().iter()
                                              .map(|ps| ps.identifier.name)
                                              .collect::<Vec<_>>();

        let containing_module;
        let last_private;
        let module = self.current_module.clone();
        match self.resolve_module_path(module,
                                       &module_path[..],
                                       UseLexicalScope,
                                       path.span,
                                       PathSearch) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared type or module `{}`",
                                          self.names_to_string(&module_path));
                        (path.span, msg)
                    }
                };

                self.resolve_error(span, &format!("failed to resolve. {}",
                                                 msg)[]);
                return None;
            }
            Indeterminate => panic!("indeterminate unexpected"),
            Success((resulting_module, resulting_last_private)) => {
                containing_module = resulting_module;
                last_private = resulting_last_private;
            }
        }

        let name = path.segments.last().unwrap().identifier.name;
        let def = match self.resolve_definition_of_name_in_module(containing_module.clone(),
                                                                  name,
                                                                  namespace) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                return None;
            }
            ChildNameDefinition(def, lp) | ImportNameDefinition(def, lp) => {
                (def, last_private.or(lp))
            }
        };
        if let Some(DefId{krate: kid, ..}) = containing_module.def_id.get() {
            self.used_crates.insert(kid);
        }
        return Some(def);
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path(&mut self,
                                   path: &Path,
                                   namespace: Namespace)
                                       -> Option<(Def, LastPrivate)> {
        let module_path = path.segments.init().iter()
                                              .map(|ps| ps.identifier.name)
                                              .collect::<Vec<_>>();

        let root_module = self.graph_root.get_module();

        let containing_module;
        let last_private;
        match self.resolve_module_path_from_root(root_module,
                                                 &module_path[..],
                                                 0,
                                                 path.span,
                                                 PathSearch,
                                                 LastMod(AllPublic)) {
            Failed(err) => {
                let (span, msg) = match err {
                    Some((span, msg)) => (span, msg),
                    None => {
                        let msg = format!("Use of undeclared module `::{}`",
                                          self.names_to_string(&module_path[..]));
                        (path.span, msg)
                    }
                };

                self.resolve_error(span, &format!("failed to resolve. {}",
                                                 msg)[]);
                return None;
            }

            Indeterminate => {
                panic!("indeterminate unexpected");
            }

            Success((resulting_module, resulting_last_private)) => {
                containing_module = resulting_module;
                last_private = resulting_last_private;
            }
        }

        let name = path.segments.last().unwrap().identifier.name;
        match self.resolve_definition_of_name_in_module(containing_module,
                                                        name,
                                                        namespace) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                return None;
            }
            ChildNameDefinition(def, lp) | ImportNameDefinition(def, lp) => {
                return Some((def, last_private.or(lp)));
            }
        }
    }

    fn resolve_identifier_in_local_ribs(&mut self,
                                        ident: Ident,
                                        namespace: Namespace,
                                        span: Span)
                                        -> Option<Def> {
        // Check the local set of ribs.
        let search_result = match namespace {
            ValueNS => {
                let renamed = mtwt::resolve(ident);
                self.search_ribs(&self.value_ribs, renamed, span)
            }
            TypeNS => {
                let name = ident.name;
                self.search_ribs(&self.type_ribs[], name, span)
            }
        };

        match search_result {
            Some(DlDef(def)) => {
                debug!("(resolving path in local ribs) resolved `{}` to \
                        local: {:?}",
                       token::get_ident(ident),
                       def);
                return Some(def);
            }
            Some(DlField) | Some(DlImpl(_)) | None => {
                return None;
            }
        }
    }

    fn resolve_item_by_name_in_lexical_scope(&mut self,
                                             name: Name,
                                             namespace: Namespace)
                                            -> Option<(Def, LastPrivate)> {
        // Check the items.
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module,
                                                 name,
                                                 namespace) {
            Success((target, _)) => {
                match (*target.bindings).def_for_namespace(namespace) {
                    None => {
                        // This can happen if we were looking for a type and
                        // found a module instead. Modules don't have defs.
                        debug!("(resolving item path by identifier in lexical \
                                 scope) failed to resolve {} after success...",
                                 token::get_name(name));
                        return None;
                    }
                    Some(def) => {
                        debug!("(resolving item path in lexical scope) \
                                resolved `{}` to item",
                               token::get_name(name));
                        // This lookup is "all public" because it only searched
                        // for one identifier in the current module (couldn't
                        // have passed through reexports or anything like that.
                        return Some((def, LastMod(AllPublic)));
                    }
                }
            }
            Indeterminate => {
                panic!("unexpected indeterminate result");
            }
            Failed(err) => {
                match err {
                    Some((span, msg)) =>
                        self.resolve_error(span, &format!("failed to resolve. {}",
                                                         msg)[]),
                    None => ()
                }

                debug!("(resolving item path by identifier in lexical scope) \
                         failed to resolve {}", token::get_name(name));
                return None;
            }
        }
    }

    fn with_no_errors<T, F>(&mut self, f: F) -> T where
        F: FnOnce(&mut Resolver) -> T,
    {
        self.emit_errors = false;
        let rs = f(self);
        self.emit_errors = true;
        rs
    }

    fn resolve_error(&self, span: Span, s: &str) {
        if self.emit_errors {
            self.session.span_err(span, s);
        }
    }

    fn find_fallback_in_self_type(&mut self, name: Name) -> FallbackSuggestion {
        fn extract_path_and_node_id(t: &Ty, allow: FallbackChecks)
                                                    -> Option<(Path, NodeId, FallbackChecks)> {
            match t.node {
                TyPath(ref path, node_id) => Some((path.clone(), node_id, allow)),
                TyPtr(ref mut_ty) => extract_path_and_node_id(&*mut_ty.ty, OnlyTraitAndStatics),
                TyRptr(_, ref mut_ty) => extract_path_and_node_id(&*mut_ty.ty, allow),
                // This doesn't handle the remaining `Ty` variants as they are not
                // that commonly the self_type, it might be interesting to provide
                // support for those in future.
                _ => None,
            }
        }

        fn get_module(this: &mut Resolver, span: Span, name_path: &[ast::Name])
                            -> Option<Rc<Module>> {
            let root = this.current_module.clone();
            let last_name = name_path.last().unwrap();

            if name_path.len() == 1 {
                match this.primitive_type_table.primitive_types.get(last_name) {
                    Some(_) => None,
                    None => {
                        match this.current_module.children.borrow().get(last_name) {
                            Some(child) => child.get_module_if_available(),
                            None => None
                        }
                    }
                }
            } else {
                match this.resolve_module_path(root,
                                                &name_path[..],
                                                UseLexicalScope,
                                                span,
                                                PathSearch) {
                    Success((module, _)) => Some(module),
                    _ => None
                }
            }
        }

        let (path, node_id, allowed) = match self.current_self_type {
            Some(ref ty) => match extract_path_and_node_id(ty, Everything) {
                Some(x) => x,
                None => return NoSuggestion,
            },
            None => return NoSuggestion,
        };

        if allowed == Everything {
            // Look for a field with the same name in the current self_type.
            match self.def_map.borrow().get(&node_id) {
                 Some(&DefTy(did, _))
                | Some(&DefStruct(did))
                | Some(&DefVariant(_, did, _)) => match self.structs.get(&did) {
                    None => {}
                    Some(fields) => {
                        if fields.iter().any(|&field_name| name == field_name) {
                            return Field;
                        }
                    }
                },
                _ => {} // Self type didn't resolve properly
            }
        }

        let name_path = path.segments.iter().map(|seg| seg.identifier.name).collect::<Vec<_>>();

        // Look for a method in the current self type's impl module.
        match get_module(self, path.span, &name_path[..]) {
            Some(module) => match module.children.borrow().get(&name) {
                Some(binding) => {
                    let p_str = self.path_names_to_string(&path);
                    match binding.def_for_namespace(ValueNS) {
                        Some(DefStaticMethod(_, provenance)) => {
                            match provenance {
                                FromImpl(_) => return StaticMethod(p_str),
                                FromTrait(_) => unreachable!()
                            }
                        }
                        Some(DefMethod(_, None, _)) if allowed == Everything => return Method,
                        Some(DefMethod(_, Some(_), _)) => return TraitItem,
                        _ => ()
                    }
                }
                None => {}
            },
            None => {}
        }

        // Look for a method in the current trait.
        match self.current_trait_ref {
            Some((did, ref trait_ref)) => {
                let path_str = self.path_names_to_string(&trait_ref.path);

                match self.trait_item_map.get(&(name, did)) {
                    Some(&StaticMethodTraitItemKind) => {
                        return TraitMethod(path_str)
                    }
                    Some(_) => return TraitItem,
                    None => {}
                }
            }
            None => {}
        }

        NoSuggestion
    }

    fn find_best_match_for_name(&mut self, name: &str, max_distance: uint)
                                -> Option<String> {
        let this = &mut *self;

        let mut maybes: Vec<token::InternedString> = Vec::new();
        let mut values: Vec<uint> = Vec::new();

        for rib in this.value_ribs.iter().rev() {
            for (&k, _) in &rib.bindings {
                maybes.push(token::get_name(k));
                values.push(usize::MAX);
            }
        }

        let mut smallest = 0;
        for (i, other) in maybes.iter().enumerate() {
            values[i] = lev_distance(name, &other);

            if values[i] <= values[smallest] {
                smallest = i;
            }
        }

        if values.len() > 0 &&
            values[smallest] != usize::MAX &&
            values[smallest] < name.len() + 2 &&
            values[smallest] <= max_distance &&
            name != &maybes[smallest][] {

            Some(maybes[smallest].to_string())

        } else {
            None
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.node {
            // The interpretation of paths depends on whether the path has
            // multiple elements in it or not.

            ExprPath(_) | ExprQPath(_) => {
                let mut path_from_qpath;
                let path = match expr.node {
                    ExprPath(ref path) => path,
                    ExprQPath(ref qpath) => {
                        self.resolve_type(&*qpath.self_type);
                        self.resolve_trait_reference(expr.id, &*qpath.trait_ref, TraitQPath);
                        path_from_qpath = qpath.trait_ref.path.clone();
                        path_from_qpath.segments.push(qpath.item_path.clone());
                        &path_from_qpath
                    }
                    _ => unreachable!()
                };
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.
                match self.resolve_path(expr.id, path, ValueNS, true) {
                    // Check if struct variant
                    Some((DefVariant(_, _, true), _)) => {
                        let path_name = self.path_names_to_string(path);
                        self.resolve_error(expr.span,
                                &format!("`{}` is a struct variant name, but \
                                          this expression \
                                          uses it like a function name",
                                         path_name));

                        self.session.span_help(expr.span,
                            &format!("Did you mean to write: \
                                     `{} {{ /* fields */ }}`?",
                                     path_name));
                    }
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving expr) resolved `{}`",
                               self.path_names_to_string(path));

                        self.record_def(expr.id, def);
                    }
                    None => {
                        // Be helpful if the name refers to a struct
                        // (The pattern matching def_tys where the id is in self.structs
                        // matches on regular structs while excluding tuple- and enum-like
                        // structs, which wouldn't result in this error.)
                        let path_name = self.path_names_to_string(path);
                        match self.with_no_errors(|this|
                            this.resolve_path(expr.id, path, TypeNS, false)) {
                            Some((DefTy(struct_id, _), _))
                              if self.structs.contains_key(&struct_id) => {
                                self.resolve_error(expr.span,
                                        &format!("`{}` is a structure name, but \
                                                  this expression \
                                                  uses it like a function name",
                                                 path_name));

                                self.session.span_help(expr.span,
                                    &format!("Did you mean to write: \
                                             `{} {{ /* fields */ }}`?",
                                             path_name));

                            }
                            _ => {
                                let mut method_scope = false;
                                self.value_ribs.iter().rev().all(|rib| {
                                    let res = match *rib {
                                        Rib { bindings: _, kind: MethodRibKind(_, _) } => true,
                                        Rib { bindings: _, kind: ItemRibKind } => false,
                                        _ => return true, // Keep advancing
                                    };

                                    method_scope = res;
                                    false // Stop advancing
                                });

                                if method_scope && &token::get_name(self.self_name)[]
                                                                   == path_name {
                                        self.resolve_error(
                                            expr.span,
                                            "`self` is not available \
                                             in a static method. Maybe a \
                                             `self` argument is missing?");
                                } else {
                                    let last_name = path.segments.last().unwrap().identifier.name;
                                    let mut msg = match self.find_fallback_in_self_type(last_name) {
                                        NoSuggestion => {
                                            // limit search to 5 to reduce the number
                                            // of stupid suggestions
                                            self.find_best_match_for_name(&path_name, 5)
                                                                .map_or("".to_string(),
                                                                        |x| format!("`{}`", x))
                                        }
                                        Field =>
                                            format!("`self.{}`", path_name),
                                        Method
                                        | TraitItem =>
                                            format!("to call `self.{}`", path_name),
                                        TraitMethod(path_str)
                                        | StaticMethod(path_str) =>
                                            format!("to call `{}::{}`", path_str, path_name)
                                    };

                                    if msg.len() > 0 {
                                        msg = format!(". Did you mean {}?", msg)
                                    }

                                    self.resolve_error(
                                        expr.span,
                                        &format!("unresolved name `{}`{}",
                                                 path_name,
                                                 msg));
                                }
                            }
                        }
                    }
                }

                visit::walk_expr(self, expr);
            }

            ExprClosure(_, ref fn_decl, ref block) => {
                self.resolve_function(ClosureRibKind(expr.id),
                                      Some(&**fn_decl), NoTypeParameters,
                                      &**block);
            }

            ExprStruct(ref path, _, _) => {
                // Resolve the path to the structure it goes to. We don't
                // check to ensure that the path is actually a structure; that
                // is checked later during typeck.
                match self.resolve_path(expr.id, path, TypeNS, false) {
                    Some(definition) => self.record_def(expr.id, definition),
                    result => {
                        debug!("(resolving expression) didn't find struct \
                                def: {:?}", result);
                        let msg = format!("`{}` does not name a structure",
                                          self.path_names_to_string(path));
                        self.resolve_error(path.span, &msg[..]);
                    }
                }

                visit::walk_expr(self, expr);
            }

            ExprLoop(_, Some(label)) | ExprWhile(_, _, Some(label)) => {
                self.with_label_rib(|this| {
                    let def_like = DlDef(DefLabel(expr.id));

                    {
                        let rib = this.label_ribs.last_mut().unwrap();
                        let renamed = mtwt::resolve(label);
                        rib.bindings.insert(renamed, def_like);
                    }

                    visit::walk_expr(this, expr);
                })
            }

            ExprBreak(Some(label)) | ExprAgain(Some(label)) => {
                let renamed = mtwt::resolve(label);
                match self.search_label(renamed) {
                    None => {
                        self.resolve_error(
                            expr.span,
                            &format!("use of undeclared label `{}`",
                                    token::get_ident(label))[])
                    }
                    Some(DlDef(def @ DefLabel(_))) => {
                        // Since this def is a label, it is never read.
                        self.record_def(expr.id, (def, LastMod(AllPublic)))
                    }
                    Some(_) => {
                        self.session.span_bug(expr.span,
                                              "label wasn't mapped to a \
                                               label def!")
                    }
                }
            }

            _ => {
                visit::walk_expr(self, expr);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &Expr) {
        match expr.node {
            ExprField(_, ident) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.search_for_traits_containing_method(ident.node.name);
                self.trait_map.insert(expr.id, traits);
            }
            ExprMethodCall(ident, _, _) => {
                debug!("(recording candidate traits for expr) recording \
                        traits for {}",
                       expr.id);
                let traits = self.search_for_traits_containing_method(ident.node.name);
                self.trait_map.insert(expr.id, traits);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn search_for_traits_containing_method(&mut self, name: Name) -> Vec<DefId> {
        debug!("(searching for traits containing method) looking for '{}'",
               token::get_name(name));

        fn add_trait_info(found_traits: &mut Vec<DefId>,
                          trait_def_id: DefId,
                          name: Name) {
            debug!("(adding trait info) found trait {}:{} for method '{}'",
                trait_def_id.krate,
                trait_def_id.node,
                token::get_name(name));
            found_traits.push(trait_def_id);
        }

        let mut found_traits = Vec::new();
        let mut search_module = self.current_module.clone();
        loop {
            // Look for the current trait.
            match self.current_trait_ref {
                Some((trait_def_id, _)) => {
                    if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
                None => {} // Nothing to do.
            }

            // Look for trait children.
            build_reduced_graph::populate_module_if_necessary(self, &search_module);

            {
                for (_, child_names) in &*search_module.children.borrow() {
                    let def = match child_names.def_for_namespace(TypeNS) {
                        Some(def) => def,
                        None => continue
                    };
                    let trait_def_id = match def {
                        DefTrait(trait_def_id) => trait_def_id,
                        _ => continue,
                    };
                    if self.trait_item_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
            }

            // Look for imports.
            for (_, import) in &*search_module.import_resolutions.borrow() {
                let target = match import.target_for_namespace(TypeNS) {
                    None => continue,
                    Some(target) => target,
                };
                let did = match target.bindings.def_for_namespace(TypeNS) {
                    Some(DefTrait(trait_def_id)) => trait_def_id,
                    Some(..) | None => continue,
                };
                if self.trait_item_map.contains_key(&(name, did)) {
                    add_trait_info(&mut found_traits, did, name);
                    let id = import.type_id;
                    self.used_imports.insert((id, TypeNS));
                    let trait_name = self.get_trait_name(did);
                    self.record_import_use(id, trait_name);
                    if let Some(DefId{krate: kid, ..}) = target.target_module.def_id.get() {
                        self.used_crates.insert(kid);
                    }
                }
            }

            match search_module.parent_link.clone() {
                NoParentLink | ModuleParentLink(..) => break,
                BlockParentLink(parent_module, _) => {
                    search_module = parent_module.upgrade().unwrap();
                }
            }
        }

        found_traits
    }

    fn record_def(&mut self, node_id: NodeId, (def, lp): (Def, LastPrivate)) {
        debug!("(recording def) recording {:?} for {}, last private {:?}",
                def, node_id, lp);
        assert!(match lp {LastImport{..} => false, _ => true},
                "Import should only be used for `use` directives");
        self.last_private.insert(node_id, lp);

        match self.def_map.borrow_mut().entry(node_id) {
            // Resolve appears to "resolve" the same ID multiple
            // times, so here is a sanity check it at least comes to
            // the same conclusion! - nmatsakis
            Occupied(entry) => if def != *entry.get() {
                self.session
                    .bug(&format!("node_id {} resolved first to {:?} and \
                                  then {:?}",
                                 node_id,
                                 *entry.get(),
                                 def)[]);
            },
            Vacant(entry) => { entry.insert(def); },
        }
    }

    fn enforce_default_binding_mode(&mut self,
                                        pat: &Pat,
                                        pat_binding_mode: BindingMode,
                                        descr: &str) {
        match pat_binding_mode {
            BindByValue(_) => {}
            BindByRef(..) => {
                self.resolve_error(pat.span,
                                   &format!("cannot use `ref` binding mode \
                                            with {}",
                                           descr)[]);
            }
        }
    }

    //
    // Diagnostics
    //
    // Diagnostics are not particularly efficient, because they're rarely
    // hit.
    //

    /// A somewhat inefficient routine to obtain the name of a module.
    fn module_to_string(&self, module: &Module) -> String {
        let mut names = Vec::new();

        fn collect_mod(names: &mut Vec<ast::Name>, module: &Module) {
            match module.parent_link {
                NoParentLink => {}
                ModuleParentLink(ref module, name) => {
                    names.push(name);
                    collect_mod(names, &*module.upgrade().unwrap());
                }
                BlockParentLink(ref module, _) => {
                    // danger, shouldn't be ident?
                    names.push(special_idents::opaque.name);
                    collect_mod(names, &*module.upgrade().unwrap());
                }
            }
        }
        collect_mod(&mut names, module);

        if names.len() == 0 {
            return "???".to_string();
        }
        self.names_to_string(&names.into_iter().rev()
                                  .collect::<Vec<ast::Name>>()[])
    }

    #[allow(dead_code)]   // useful for debugging
    fn dump_module(&mut self, module_: Rc<Module>) {
        debug!("Dump of module `{}`:", self.module_to_string(&*module_));

        debug!("Children:");
        build_reduced_graph::populate_module_if_necessary(self, &module_);
        for (&name, _) in &*module_.children.borrow() {
            debug!("* {}", token::get_name(name));
        }

        debug!("Import resolutions:");
        let import_resolutions = module_.import_resolutions.borrow();
        for (&name, import_resolution) in &*import_resolutions {
            let value_repr;
            match import_resolution.target_for_namespace(ValueNS) {
                None => { value_repr = "".to_string(); }
                Some(_) => {
                    value_repr = " value:?".to_string();
                    // FIXME #4954
                }
            }

            let type_repr;
            match import_resolution.target_for_namespace(TypeNS) {
                None => { type_repr = "".to_string(); }
                Some(_) => {
                    type_repr = " type:?".to_string();
                    // FIXME #4954
                }
            }

            debug!("* {}:{}{}", token::get_name(name), value_repr, type_repr);
        }
    }
}

pub struct CrateMap {
    pub def_map: DefMap,
    pub freevars: RefCell<FreevarMap>,
    pub export_map: ExportMap,
    pub trait_map: TraitMap,
    pub external_exports: ExternalExports,
    pub last_private_map: LastPrivateMap,
    pub glob_map: Option<GlobMap>
}

#[derive(PartialEq,Copy)]
pub enum MakeGlobMap {
    Yes,
    No
}

/// Entry point to crate resolution.
pub fn resolve_crate<'a, 'tcx>(session: &'a Session,
                               ast_map: &'a ast_map::Map<'tcx>,
                               _: &LanguageItems,
                               krate: &Crate,
                               make_glob_map: MakeGlobMap)
                               -> CrateMap {
    let mut resolver = Resolver::new(session, ast_map, krate.span, make_glob_map);

    build_reduced_graph::build_reduced_graph(&mut resolver, krate);
    session.abort_if_errors();

    resolver.resolve_imports();
    session.abort_if_errors();

    record_exports::record(&mut resolver);
    session.abort_if_errors();

    resolver.resolve_crate(krate);
    session.abort_if_errors();

    check_unused::check_crate(&mut resolver, krate);

    CrateMap {
        def_map: resolver.def_map,
        freevars: resolver.freevars,
        export_map: resolver.export_map,
        trait_map: resolver.trait_map,
        external_exports: resolver.external_exports,
        last_private_map: resolver.last_private,
        glob_map: if resolver.make_glob_map {
                        Some(resolver.glob_map)
                    } else {
                        None
                    },
    }
}
