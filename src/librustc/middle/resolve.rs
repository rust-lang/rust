// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(non_camel_case_types)]

use driver::session::Session;
use metadata::csearch;
use metadata::decoder::{DefLike, DlDef, DlField, DlImpl};
use middle::lang_items::LanguageItems;
use middle::lint::{UnnecessaryQualification, UnusedImports};
use middle::pat_util::pat_bindings;
use util::nodemap::{NodeMap, DefIdSet, FnvHashMap};

use syntax::ast::*;
use syntax::ast;
use syntax::ast_util::{def_id_of_def, local_def};
use syntax::ast_util::{path_to_ident, walk_pat, trait_method_to_ty_method};
use syntax::ext::mtwt;
use syntax::parse::token::special_idents;
use syntax::parse::token;
use syntax::print::pprust::path_to_str;
use syntax::codemap::{Span, DUMMY_SP, Pos};
use syntax::owned_slice::OwnedSlice;
use syntax::visit;
use syntax::visit::Visitor;

use collections::{HashMap, HashSet};
use std::cell::{Cell, RefCell};
use std::mem::replace;
use std::rc::{Rc, Weak};
use std::strbuf::StrBuf;
use std::uint;

// Definition mapping
pub type DefMap = RefCell<NodeMap<Def>>;

struct binding_info {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<Name,binding_info>;

// Trait method resolution
pub type TraitMap = NodeMap<Vec<DefId> >;

// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap2 = RefCell<NodeMap<Vec<Export2> >>;

pub struct Export2 {
    pub name: StrBuf,        // The name of the target.
    pub def_id: DefId,     // The definition of the target.
}

// This set contains all exported definitions from external crates. The set does
// not contain any entries from local crates.
pub type ExternalExports = DefIdSet;

// FIXME: dox
pub type LastPrivateMap = NodeMap<LastPrivate>;

pub enum LastPrivate {
    LastMod(PrivateDep),
    // `use` directives (imports) can refer to two separate definitions in the
    // type and value namespaces. We record here the last private node for each
    // and whether the import is in fact used for each.
    // If the Option<PrivateDep> fields are None, it means there is no definition
    // in that namespace.
    LastImport{pub value_priv: Option<PrivateDep>,
               pub value_used: ImportUse,
               pub type_priv: Option<PrivateDep>,
               pub type_used: ImportUse},
}

pub enum PrivateDep {
    AllPublic,
    DependsOn(DefId),
}

// How an import is used.
#[deriving(Eq)]
pub enum ImportUse {
    Unused,       // The import is not used.
    Used,         // The import is used.
}

impl LastPrivate {
    fn or(self, other: LastPrivate) -> LastPrivate {
        match (self, other) {
            (me, LastMod(AllPublic)) => me,
            (_, other) => other,
        }
    }
}

#[deriving(Eq)]
enum PatternBindingMode {
    RefutableMode,
    LocalIrrefutableMode,
    ArgumentIrrefutableMode,
}

#[deriving(Eq, TotalEq, Hash)]
enum Namespace {
    TypeNS,
    ValueNS
}

#[deriving(Eq)]
enum NamespaceError {
    NoError,
    ModuleError,
    TypeError,
    ValueError
}

/// A NamespaceResult represents the result of resolving an import in
/// a particular namespace. The result is either definitely-resolved,
/// definitely- unresolved, or unknown.
#[deriving(Clone)]
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

impl<'a> Visitor<()> for Resolver<'a> {
    fn visit_item(&mut self, item: &Item, _: ()) {
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm: &Arm, _: ()) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block: &Block, _: ()) {
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr: &Expr, _: ()) {
        self.resolve_expr(expr);
    }
    fn visit_local(&mut self, local: &Local, _: ()) {
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty: &Ty, _: ()) {
        self.resolve_type(ty);
    }
}

/// Contains data for specific types of import directives.
enum ImportDirectiveSubclass {
    SingleImport(Ident /* target */, Ident /* source */),
    GlobImport
}

/// The context that we thread through while building the reduced graph.
#[deriving(Clone)]
enum ReducedGraphParent {
    ModuleReducedGraphParent(Rc<Module>)
}

impl ReducedGraphParent {
    fn module(&self) -> Rc<Module> {
        match *self {
            ModuleReducedGraphParent(ref m) => {
                m.clone()
            }
        }
    }
}

enum ResolveResult<T> {
    Failed,         // Failed to resolve the name.
    Indeterminate,  // Couldn't determine due to unresolved globs.
    Success(T)      // Successfully resolved the import.
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
    TraitMethod,
    StaticMethod(StrBuf),
    StaticTraitMethod(StrBuf),
}

enum TypeParameters<'a> {
    NoTypeParameters,                   //< No type parameters.
    HasTypeParameters(&'a Generics,  //< Type parameters.
                      NodeId,          //< ID of the enclosing item

                      // The index to start numbering the type parameters at.
                      // This is zero if this is the outermost set of type
                      // parameters, or equal to the number of outer type
                      // parameters. For example, if we have:
                      //
                      //   impl I<T> {
                      //     fn method<U>() { ... }
                      //   }
                      //
                      // The index at the method site will be 1, because the
                      // outer T had index 0.
                      uint,

                      // The kind of the rib used for type parameters.
                      RibKind)
}

// The rib kind controls the translation of argument or local definitions
// (`def_arg` or `def_local`) to upvars (`def_upvar`).

enum RibKind {
    // No translation needs to be applied.
    NormalRibKind,

    // We passed through a function scope at the given node ID. Translate
    // upvars as appropriate.
    FunctionRibKind(NodeId /* func id */, NodeId /* body id */),

    // We passed through an impl or trait and are now in one of its
    // methods. Allow references to ty params that impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
              // parent;   method itself
    MethodRibKind(NodeId, MethodSort),

    // We passed through a function *item* scope. Disallow upvars.
    OpaqueFunctionRibKind,

    // We're in a constant item. Can't refer to dynamic stuff.
    ConstantItemRibKind
}

// Methods can be required or provided. Required methods only occur in traits.
enum MethodSort {
    Required,
    Provided(NodeId)
}

enum UseLexicalScopeFlag {
    DontUseLexicalScope,
    UseLexicalScope
}

enum ModulePrefixResult {
    NoPrefixFound,
    PrefixFound(Rc<Module>, uint)
}

#[deriving(Eq)]
enum NameSearchType {
    /// We're doing a name search in order to resolve a `use` directive.
    ImportSearch,

    /// We're doing a name search in order to resolve a path type, a path
    /// expression, or a path pattern.
    PathSearch,
}

enum BareIdentifierPatternResolution {
    FoundStructOrEnumVariant(Def, LastPrivate),
    FoundConst(Def, LastPrivate),
    BareIdentifierPatternUnresolved
}

// Specifies how duplicates should be handled when adding a child item if
// another item exists with the same name in some namespace.
#[deriving(Eq)]
enum DuplicateCheckingMode {
    ForbidDuplicateModules,
    ForbidDuplicateTypes,
    ForbidDuplicateValues,
    ForbidDuplicateTypesAndValues,
    OverwriteDuplicates
}

/// One local scope.
struct Rib {
    bindings: RefCell<HashMap<Name, DefLike>>,
    kind: RibKind,
}

impl Rib {
    fn new(kind: RibKind) -> Rib {
        Rib {
            bindings: RefCell::new(HashMap::new()),
            kind: kind
        }
    }
}

/// One import directive.
struct ImportDirective {
    module_path: Vec<Ident>,
    subclass: ImportDirectiveSubclass,
    span: Span,
    id: NodeId,
    is_public: bool, // see note in ImportResolution about how to use this
}

impl ImportDirective {
    fn new(module_path: Vec<Ident> ,
           subclass: ImportDirectiveSubclass,
           span: Span,
           id: NodeId,
           is_public: bool)
           -> ImportDirective {
        ImportDirective {
            module_path: module_path,
            subclass: subclass,
            span: span,
            id: id,
            is_public: is_public,
        }
    }
}

/// The item that an import resolves to.
#[deriving(Clone)]
struct Target {
    target_module: Rc<Module>,
    bindings: Rc<NameBindings>,
}

impl Target {
    fn new(target_module: Rc<Module>, bindings: Rc<NameBindings>) -> Target {
        Target {
            target_module: target_module,
            bindings: bindings
        }
    }
}

/// An ImportResolution represents a particular `use` directive.
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
}

/// The link from a module up to its nearest parent node.
#[deriving(Clone)]
enum ParentLink {
    NoParentLink,
    ModuleParentLink(Weak<Module>, Ident),
    BlockParentLink(Weak<Module>, NodeId)
}

/// The type of module this is.
#[deriving(Eq)]
enum ModuleKind {
    NormalModuleKind,
    ExternModuleKind,
    TraitModuleKind,
    ImplModuleKind,
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
            anonymous_children: RefCell::new(NodeMap::new()),
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

// Records a possibly-private type definition.
#[deriving(Clone)]
struct TypeNsDef {
    is_public: bool, // see note in ImportResolution about how to use this
    module_def: Option<Rc<Module>>,
    type_def: Option<Def>,
    type_span: Option<Span>
}

// Records a possibly-private value definition.
#[deriving(Clone)]
struct ValueNsDef {
    is_public: bool, // see note in ImportResolution about how to use this
    def: Def,
    value_span: Option<Span>,
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
struct NameBindings {
    type_def: RefCell<Option<TypeNsDef>>,   //< Meaning in type namespace.
    value_def: RefCell<Option<ValueNsDef>>, //< Meaning in value namespace.
}

/// Ways in which a trait can be referenced
enum TraitReferenceType {
    TraitImplementation,             // impl SomeTrait for T { ... }
    TraitDerivation,                 // trait T : SomeTrait { ... }
    TraitBoundingTypeParameter,      // fn f<T:SomeTrait>() { ... }
}

impl NameBindings {
    /// Creates a new module in this set of name bindings.
    fn define_module(&self,
                     parent_link: ParentLink,
                     def_id: Option<DefId>,
                     kind: ModuleKind,
                     external: bool,
                     is_public: bool,
                     sp: Span) {
        // Merges the module with the existing type def or creates a new one.
        let module_ = Rc::new(Module::new(parent_link, def_id, kind, external,
                                          is_public));
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    is_public: is_public,
                    module_def: Some(module_),
                    type_def: None,
                    type_span: Some(sp)
                });
            }
            Some(type_def) => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    is_public: is_public,
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
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                let module = Module::new(parent_link, def_id, kind,
                                         external, is_public);
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    is_public: is_public,
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
                            is_public: is_public,
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
    fn define_type(&self, def: Def, sp: Span, is_public: bool) {
        // Merges the type with the existing type def or creates a new one.
        let type_def = self.type_def.borrow().clone();
        match type_def {
            None => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    module_def: None,
                    type_def: Some(def),
                    type_span: Some(sp),
                    is_public: is_public,
                });
            }
            Some(type_def) => {
                *self.type_def.borrow_mut() = Some(TypeNsDef {
                    type_def: Some(def),
                    type_span: Some(sp),
                    module_def: type_def.module_def,
                    is_public: is_public,
                });
            }
        }
    }

    /// Records a value definition.
    fn define_value(&self, def: Def, sp: Span, is_public: bool) {
        *self.value_def.borrow_mut() = Some(ValueNsDef {
            def: def,
            value_span: Some(sp),
            is_public: is_public,
        });
    }

    /// Returns the module node if applicable.
    fn get_module_if_available(&self) -> Option<Rc<Module>> {
        match *self.type_def.borrow() {
            Some(ref type_def) => type_def.module_def.clone(),
            None => None
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module(&self) -> Rc<Module> {
        match self.get_module_if_available() {
            None => {
                fail!("get_module called on a node with no module \
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
        match namespace {
            TypeNS => match *self.type_def.borrow() {
                Some(ref def) => def.is_public, None => false
            },
            ValueNS => match *self.value_def.borrow() {
                Some(ref def) => def.is_public, None => false
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

fn NameBindings() -> NameBindings {
    NameBindings {
        type_def: RefCell::new(None),
        value_def: RefCell::new(None),
    }
}

/// Interns the names of the primitive types.
struct PrimitiveTypeTable {
    primitive_types: HashMap<Name, PrimTy>,
}

impl PrimitiveTypeTable {
    fn intern(&mut self, string: &str, primitive_type: PrimTy) {
        self.primitive_types.insert(token::intern(string), primitive_type);
    }
}

fn PrimitiveTypeTable() -> PrimitiveTypeTable {
    let mut table = PrimitiveTypeTable {
        primitive_types: HashMap::new()
    };

    table.intern("bool",    TyBool);
    table.intern("char",    TyChar);
    table.intern("f32",     TyFloat(TyF32));
    table.intern("f64",     TyFloat(TyF64));
    table.intern("f128",    TyFloat(TyF128));
    table.intern("int",     TyInt(TyI));
    table.intern("i8",      TyInt(TyI8));
    table.intern("i16",     TyInt(TyI16));
    table.intern("i32",     TyInt(TyI32));
    table.intern("i64",     TyInt(TyI64));
    table.intern("str",     TyStr);
    table.intern("uint",    TyUint(TyU));
    table.intern("u8",      TyUint(TyU8));
    table.intern("u16",     TyUint(TyU16));
    table.intern("u32",     TyUint(TyU32));
    table.intern("u64",     TyUint(TyU64));

    return table;
}


fn namespace_error_to_str(ns: NamespaceError) -> &'static str {
    match ns {
        NoError     => "",
        ModuleError => "module",
        TypeError   => "type",
        ValueError  => "value",
    }
}

fn Resolver<'a>(session: &'a Session,
                lang_items: &'a LanguageItems,
                crate_span: Span) -> Resolver<'a> {
    let graph_root = NameBindings();

    graph_root.define_module(NoParentLink,
                             Some(DefId { krate: 0, node: 0 }),
                             NormalModuleKind,
                             false,
                             true,
                             crate_span);

    let current_module = graph_root.get_module();

    let this = Resolver {
        session: session,
        lang_items: lang_items,

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        graph_root: graph_root,

        method_map: RefCell::new(FnvHashMap::new()),
        structs: FnvHashMap::new(),

        unresolved_imports: 0,

        current_module: current_module,
        value_ribs: RefCell::new(Vec::new()),
        type_ribs: RefCell::new(Vec::new()),
        label_ribs: RefCell::new(Vec::new()),

        current_trait_ref: None,
        current_self_type: None,

        self_ident: special_idents::self_,
        type_self_ident: special_idents::type_self,

        primitive_type_table: PrimitiveTypeTable(),

        namespaces: vec!(TypeNS, ValueNS),

        def_map: RefCell::new(NodeMap::new()),
        export_map2: RefCell::new(NodeMap::new()),
        trait_map: NodeMap::new(),
        used_imports: HashSet::new(),
        external_exports: DefIdSet::new(),
        last_private: NodeMap::new(),

        emit_errors: true,
    };

    this
}

/// The main resolver class.
struct Resolver<'a> {
    session: &'a Session,
    lang_items: &'a LanguageItems,

    graph_root: NameBindings,

    method_map: RefCell<FnvHashMap<(Name, DefId), ast::ExplicitSelf_>>,
    structs: FnvHashMap<DefId, Vec<Name>>,

    // The number of imports that are currently unresolved.
    unresolved_imports: uint,

    // The module that represents the current item scope.
    current_module: Rc<Module>,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: RefCell<Vec<Rib>>,

    // The current set of local scopes, for types.
    type_ribs: RefCell<Vec<Rib>>,

    // The current set of local scopes, for labels.
    label_ribs: RefCell<Vec<Rib>>,

    // The trait that the current context can refer to.
    current_trait_ref: Option<(DefId, TraitRef)>,

    // The current self type if inside an impl (used for better errors).
    current_self_type: Option<Ty>,

    // The ident for the keyword "self".
    self_ident: Ident,
    // The ident for the non-keyword "Self".
    type_self_ident: Ident,

    // The idents for the primitive types.
    primitive_type_table: PrimitiveTypeTable,

    // The four namespaces.
    namespaces: Vec<Namespace> ,

    def_map: DefMap,
    export_map2: ExportMap2,
    trait_map: TraitMap,
    external_exports: ExternalExports,
    last_private: LastPrivateMap,

    // Whether or not to print error messages. Can be set to true
    // when getting additional info for error message suggestions,
    // so as to avoid printing duplicate errors
    emit_errors: bool,

    used_imports: HashSet<(NodeId, Namespace)>,
}

struct BuildReducedGraphVisitor<'a, 'b> {
    resolver: &'a mut Resolver<'b>,
}

impl<'a, 'b> Visitor<ReducedGraphParent> for BuildReducedGraphVisitor<'a, 'b> {

    fn visit_item(&mut self, item: &Item, context: ReducedGraphParent) {
        let p = self.resolver.build_reduced_graph_for_item(item, context);
        visit::walk_item(self, item, p);
    }

    fn visit_foreign_item(&mut self, foreign_item: &ForeignItem,
                          context: ReducedGraphParent) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item,
                                                           context.clone(),
                                                           |r| {
            let mut v = BuildReducedGraphVisitor{ resolver: r };
            visit::walk_foreign_item(&mut v, foreign_item, context.clone());
        })
    }

    fn visit_view_item(&mut self, view_item: &ViewItem, context: ReducedGraphParent) {
        self.resolver.build_reduced_graph_for_view_item(view_item, context);
    }

    fn visit_block(&mut self, block: &Block, context: ReducedGraphParent) {
        let np = self.resolver.build_reduced_graph_for_block(block, context);
        visit::walk_block(self, block, np);
    }

}

struct UnusedImportCheckVisitor<'a, 'b> { resolver: &'a mut Resolver<'b> }

impl<'a, 'b> Visitor<()> for UnusedImportCheckVisitor<'a, 'b> {
    fn visit_view_item(&mut self, vi: &ViewItem, _: ()) {
        self.resolver.check_for_item_unused_imports(vi);
        visit::walk_view_item(self, vi, ());
    }
}

impl<'a> Resolver<'a> {
    /// The main name resolution procedure.
    fn resolve(&mut self, krate: &ast::Crate) {
        self.build_reduced_graph(krate);
        self.session.abort_if_errors();

        self.resolve_imports();
        self.session.abort_if_errors();

        self.record_exports();
        self.session.abort_if_errors();

        self.resolve_crate(krate);
        self.session.abort_if_errors();

        self.check_for_unused_imports(krate);
    }

    //
    // Reduced graph building
    //
    // Here we build the "reduced graph": the graph of the module tree without
    // any imports resolved.
    //

    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(&mut self, krate: &ast::Crate) {
        let initial_parent =
            ModuleReducedGraphParent(self.graph_root.get_module());

        let mut visitor = BuildReducedGraphVisitor { resolver: self, };
        visit::walk_crate(&mut visitor, krate, initial_parent);
    }

    /**
     * Adds a new child item to the module definition of the parent node and
     * returns its corresponding name bindings as well as the current parent.
     * Or, if we're inside a block, creates (or reuses) an anonymous module
     * corresponding to the innermost block ID and returns the name bindings
     * as well as the newly-created parent.
     *
     * If this node does not have a module definition and we are not inside
     * a block, fails.
     */
    fn add_child(&self,
                 name: Ident,
                 reduced_graph_parent: ReducedGraphParent,
                 duplicate_checking_mode: DuplicateCheckingMode,
                 // For printing errors
                 sp: Span)
                 -> Rc<NameBindings> {
        // If this is the immediate descendant of a module, then we add the
        // child name directly. Otherwise, we create or reuse an anonymous
        // module and add the child to that.

        let module_ = reduced_graph_parent.module();

        // Add or reuse the child.
        let child = module_.children.borrow().find_copy(&name.name);
        match child {
            None => {
                let child = Rc::new(NameBindings());
                module_.children.borrow_mut().insert(name.name, child.clone());
                child
            }
            Some(child) => {
                // Enforce the duplicate checking mode:
                //
                // * If we're requesting duplicate module checking, check that
                //   there isn't a module in the module with the same name.
                //
                // * If we're requesting duplicate type checking, check that
                //   there isn't a type in the module with the same name.
                //
                // * If we're requesting duplicate value checking, check that
                //   there isn't a value in the module with the same name.
                //
                // * If we're requesting duplicate type checking and duplicate
                //   value checking, check that there isn't a duplicate type
                //   and a duplicate value with the same name.
                //
                // * If no duplicate checking was requested at all, do
                //   nothing.

                let mut duplicate_type = NoError;
                let ns = match duplicate_checking_mode {
                    ForbidDuplicateModules => {
                        if child.get_module_if_available().is_some() {
                            duplicate_type = ModuleError;
                        }
                        Some(TypeNS)
                    }
                    ForbidDuplicateTypes => {
                        match child.def_for_namespace(TypeNS) {
                            Some(DefMod(_)) | None => {}
                            Some(_) => duplicate_type = TypeError
                        }
                        Some(TypeNS)
                    }
                    ForbidDuplicateValues => {
                        if child.defined_in_namespace(ValueNS) {
                            duplicate_type = ValueError;
                        }
                        Some(ValueNS)
                    }
                    ForbidDuplicateTypesAndValues => {
                        let mut n = None;
                        match child.def_for_namespace(TypeNS) {
                            Some(DefMod(_)) | None => {}
                            Some(_) => {
                                n = Some(TypeNS);
                                duplicate_type = TypeError;
                            }
                        };
                        if child.defined_in_namespace(ValueNS) {
                            duplicate_type = ValueError;
                            n = Some(ValueNS);
                        }
                        n
                    }
                    OverwriteDuplicates => None
                };
                if duplicate_type != NoError {
                    // Return an error here by looking up the namespace that
                    // had the duplicate.
                    let ns = ns.unwrap();
                    self.resolve_error(sp,
                        format!("duplicate definition of {} `{}`",
                             namespace_error_to_str(duplicate_type),
                             token::get_ident(name)));
                    {
                        let r = child.span_for_namespace(ns);
                        for sp in r.iter() {
                            self.session.span_note(*sp,
                                 format!("first definition of {} `{}` here",
                                      namespace_error_to_str(duplicate_type),
                                      token::get_ident(name)));
                        }
                    }
                }
                child
            }
        }
    }

    fn block_needs_anonymous_module(&mut self, block: &Block) -> bool {
        // If the block has view items, we need an anonymous module.
        if block.view_items.len() > 0 {
            return true;
        }

        // Check each statement.
        for statement in block.stmts.iter() {
            match statement.node {
                StmtDecl(declaration, _) => {
                    match declaration.node {
                        DeclItem(_) => {
                            return true;
                        }
                        _ => {
                            // Keep searching.
                        }
                    }
                }
                _ => {
                    // Keep searching.
                }
            }
        }

        // If we found neither view items nor items, we don't need to create
        // an anonymous module.

        return false;
    }

    fn get_parent_link(&mut self, parent: ReducedGraphParent, name: Ident)
                           -> ParentLink {
        match parent {
            ModuleReducedGraphParent(module_) => {
                return ModuleParentLink(module_.downgrade(), name);
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self,
                                    item: &Item,
                                    parent: ReducedGraphParent)
                                    -> ReducedGraphParent
    {
        let ident = item.ident;
        let sp = item.span;
        let is_public = item.vis == ast::Public;

        match item.node {
            ItemMod(..) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateModules, sp);

                let parent_link = self.get_parent_link(parent, ident);
                let def_id = DefId { krate: 0, node: item.id };
                name_bindings.define_module(parent_link,
                                            Some(def_id),
                                            NormalModuleKind,
                                            false,
                                            item.vis == ast::Public,
                                            sp);

                ModuleReducedGraphParent(name_bindings.get_module())
            }

            ItemForeignMod(..) => parent,

            // These items live in the value namespace.
            ItemStatic(_, m, _) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateValues, sp);
                let mutbl = m == ast::MutMutable;

                name_bindings.define_value
                    (DefStatic(local_def(item.id), mutbl), sp, is_public);
                parent
            }
            ItemFn(_, fn_style, _, _, _) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateValues, sp);

                let def = DefFn(local_def(item.id), fn_style);
                name_bindings.define_value(def, sp, is_public);
                parent
            }

            // These items live in the type namespace.
            ItemTy(..) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (DefTy(local_def(item.id)), sp, is_public);
                parent
            }

            ItemEnum(ref enum_definition, _) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (DefTy(local_def(item.id)), sp, is_public);

                for &variant in (*enum_definition).variants.iter() {
                    self.build_reduced_graph_for_variant(
                        variant,
                        local_def(item.id),
                        parent.clone(),
                        is_public);
                }
                parent
            }

            // These items live in both the type and value namespaces.
            ItemStruct(struct_def, _) => {
                // Adding to both Type and Value namespaces or just Type?
                let (forbid, ctor_id) = match struct_def.ctor_id {
                    Some(ctor_id)   => (ForbidDuplicateTypesAndValues, Some(ctor_id)),
                    None            => (ForbidDuplicateTypes, None)
                };

                let name_bindings = self.add_child(ident, parent.clone(), forbid, sp);

                // Define a name in the type namespace.
                name_bindings.define_type(DefTy(local_def(item.id)), sp, is_public);

                // If this is a newtype or unit-like struct, define a name
                // in the value namespace as well
                ctor_id.while_some(|cid| {
                    name_bindings.define_value(DefStruct(local_def(cid)), sp,
                                               is_public);
                    None
                });

                // Record the def ID and fields of this struct.
                let named_fields = struct_def.fields.iter().filter_map(|f| {
                    match f.node.kind {
                        NamedField(ident, _) => Some(ident.name),
                        UnnamedField(_) => None
                    }
                }).collect();
                self.structs.insert(local_def(item.id), named_fields);

                parent
            }

            ItemImpl(_, None, ty, ref methods) => {
                // If this implements an anonymous trait, then add all the
                // methods within to a new module, if the type was defined
                // within this module.
                //
                // FIXME (#3785): This is quite unsatisfactory. Perhaps we
                // should modify anonymous traits to only be implementable in
                // the same module that declared the type.

                // Create the module and add all methods.
                match ty.node {
                    TyPath(ref path, _, _) if path.segments.len() == 1 => {
                        let name = path_to_ident(path);

                        let parent_opt = parent.module().children.borrow()
                                               .find_copy(&name.name);
                        let new_parent = match parent_opt {
                            // It already exists
                            Some(ref child) if child.get_module_if_available()
                                                .is_some() &&
                                           child.get_module().kind.get() ==
                                                ImplModuleKind => {
                                ModuleReducedGraphParent(child.get_module())
                            }
                            // Create the module
                            _ => {
                                let name_bindings =
                                    self.add_child(name,
                                                   parent.clone(),
                                                   ForbidDuplicateModules,
                                                   sp);

                                let parent_link =
                                    self.get_parent_link(parent.clone(), ident);
                                let def_id = local_def(item.id);
                                let ns = TypeNS;
                                let is_public =
                                    !name_bindings.defined_in_namespace(ns) ||
                                     name_bindings.defined_in_public_namespace(ns);

                                name_bindings.define_module(parent_link,
                                                            Some(def_id),
                                                            ImplModuleKind,
                                                            false,
                                                            is_public,
                                                            sp);

                                ModuleReducedGraphParent(
                                    name_bindings.get_module())
                            }
                        };

                        // For each method...
                        for method in methods.iter() {
                            // Add the method to the module.
                            let ident = method.ident;
                            let method_name_bindings =
                                self.add_child(ident,
                                               new_parent.clone(),
                                               ForbidDuplicateValues,
                                               method.span);
                            let def = match method.explicit_self.node {
                                SelfStatic => {
                                    // Static methods become
                                    // `def_static_method`s.
                                    DefStaticMethod(local_def(method.id),
                                                      FromImpl(local_def(
                                                        item.id)),
                                                      method.fn_style)
                                }
                                _ => {
                                    // Non-static methods become
                                    // `def_method`s.
                                    DefMethod(local_def(method.id), None)
                                }
                            };

                            let is_public = method.vis == ast::Public;
                            method_name_bindings.define_value(def,
                                                              method.span,
                                                              is_public);
                        }
                    }
                    _ => {}
                }

                parent
            }

            ItemImpl(_, Some(_), _, _) => parent,

            ItemTrait(_, _, _, ref methods) => {
                let name_bindings =
                    self.add_child(ident, parent.clone(), ForbidDuplicateTypes, sp);

                // Add all the methods within to a new module.
                let parent_link = self.get_parent_link(parent.clone(), ident);
                name_bindings.define_module(parent_link,
                                            Some(local_def(item.id)),
                                            TraitModuleKind,
                                            false,
                                            item.vis == ast::Public,
                                            sp);
                let module_parent = ModuleReducedGraphParent(name_bindings.
                                                             get_module());

                let def_id = local_def(item.id);

                // Add the names of all the methods to the trait info.
                for method in methods.iter() {
                    let ty_m = trait_method_to_ty_method(method);

                    let ident = ty_m.ident;

                    // Add it as a name in the trait module.
                    let def = match ty_m.explicit_self.node {
                        SelfStatic => {
                            // Static methods become `def_static_method`s.
                            DefStaticMethod(local_def(ty_m.id),
                                              FromTrait(local_def(item.id)),
                                              ty_m.fn_style)
                        }
                        _ => {
                            // Non-static methods become `def_method`s.
                            DefMethod(local_def(ty_m.id),
                                       Some(local_def(item.id)))
                        }
                    };

                    let method_name_bindings =
                        self.add_child(ident,
                                       module_parent.clone(),
                                       ForbidDuplicateValues,
                                       ty_m.span);
                    method_name_bindings.define_value(def, ty_m.span, true);

                    self.method_map.borrow_mut().insert((ident.name, def_id),
                                                        ty_m.explicit_self.node);
                }

                name_bindings.define_type(DefTrait(def_id), sp, is_public);
                parent
            }
            ItemMac(..) => parent
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and/or value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &Variant,
                                       item_id: DefId,
                                       parent: ReducedGraphParent,
                                       is_public: bool) {
        let ident = variant.node.name;

        match variant.node.kind {
            TupleVariantKind(_) => {
                let child = self.add_child(ident, parent, ForbidDuplicateValues, variant.span);
                child.define_value(DefVariant(item_id,
                                              local_def(variant.node.id), false),
                                   variant.span, is_public);
            }
            StructVariantKind(_) => {
                let child = self.add_child(ident, parent,
                                           ForbidDuplicateTypesAndValues,
                                           variant.span);
                child.define_type(DefVariant(item_id,
                                             local_def(variant.node.id), true),
                                  variant.span, is_public);

                // Not adding fields for variants as they are not accessed with a self receiver
                self.structs.insert(local_def(variant.node.id), Vec::new());
            }
        }
    }

    /// Constructs the reduced graph for one 'view item'. View items consist
    /// of imports and use directives.
    fn build_reduced_graph_for_view_item(&mut self, view_item: &ViewItem,
                                         parent: ReducedGraphParent) {
        match view_item.node {
            ViewItemUse(ref view_path) => {
                // Extract and intern the module part of the path. For
                // globs and lists, the path is found directly in the AST;
                // for simple paths we have to munge the path a little.

                let mut module_path = Vec::new();
                match view_path.node {
                    ViewPathSimple(_, ref full_path, _) => {
                        let path_len = full_path.segments.len();
                        assert!(path_len != 0);

                        for (i, segment) in full_path.segments
                                                     .iter()
                                                     .enumerate() {
                            if i != path_len - 1 {
                                module_path.push(segment.identifier)
                            }
                        }
                    }

                    ViewPathGlob(ref module_ident_path, _) |
                    ViewPathList(ref module_ident_path, _, _) => {
                        for segment in module_ident_path.segments.iter() {
                            module_path.push(segment.identifier)
                        }
                    }
                }

                // Build up the import directives.
                let module_ = parent.module();
                let is_public = view_item.vis == ast::Public;
                match view_path.node {
                    ViewPathSimple(binding, ref full_path, id) => {
                        let source_ident =
                            full_path.segments.last().unwrap().identifier;
                        let subclass = SingleImport(binding,
                                                    source_ident);
                        self.build_import_directive(&*module_,
                                                    module_path,
                                                    subclass,
                                                    view_path.span,
                                                    id,
                                                    is_public);
                    }
                    ViewPathList(_, ref source_idents, _) => {
                        for source_ident in source_idents.iter() {
                            let name = source_ident.node.name;
                            self.build_import_directive(
                                &*module_,
                                module_path.clone(),
                                SingleImport(name, name),
                                source_ident.span,
                                source_ident.node.id,
                                is_public);
                        }
                    }
                    ViewPathGlob(_, id) => {
                        self.build_import_directive(&*module_,
                                                    module_path,
                                                    GlobImport,
                                                    view_path.span,
                                                    id,
                                                    is_public);
                    }
                }
            }

            ViewItemExternCrate(name, _, node_id) => {
                // n.b. we don't need to look at the path option here, because cstore already did
                match self.session.cstore.find_extern_mod_stmt_cnum(node_id) {
                    Some(crate_id) => {
                        let def_id = DefId { krate: crate_id, node: 0 };
                        self.external_exports.insert(def_id);
                        let parent_link = ModuleParentLink
                            (parent.module().downgrade(), name);
                        let external_module = Rc::new(Module::new(parent_link,
                                                                  Some(def_id),
                                                                  NormalModuleKind,
                                                                  false,
                                                                  true));

                        parent.module().external_module_children
                              .borrow_mut().insert(name.name,
                                                   external_module.clone());

                        self.build_reduced_graph_for_external_crate(
                            external_module);
                    }
                    None => {}  // Ignore.
                }
            }
        }
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(&mut self,
                                            foreign_item: &ForeignItem,
                                            parent: ReducedGraphParent,
                                            f: |&mut Resolver|) {
        let name = foreign_item.ident;
        let is_public = foreign_item.vis == ast::Public;
        let name_bindings =
            self.add_child(name, parent, ForbidDuplicateValues,
                           foreign_item.span);

        match foreign_item.node {
            ForeignItemFn(_, ref generics) => {
                let def = DefFn(local_def(foreign_item.id), UnsafeFn);
                name_bindings.define_value(def, foreign_item.span, is_public);

                self.with_type_parameter_rib(
                    HasTypeParameters(generics,
                                      foreign_item.id,
                                      0,
                                      NormalRibKind),
                    f);
            }
            ForeignItemStatic(_, m) => {
                let def = DefStatic(local_def(foreign_item.id), m);
                name_bindings.define_value(def, foreign_item.span, is_public);

                f(self)
            }
        }
    }

    fn build_reduced_graph_for_block(&mut self,
                                         block: &Block,
                                         parent: ReducedGraphParent)
                                            -> ReducedGraphParent
    {
        if self.block_needs_anonymous_module(block) {
            let block_id = block.id;

            debug!("(building reduced graph for block) creating a new \
                    anonymous module for block {}",
                   block_id);

            let parent_module = parent.module();
            let new_module = Rc::new(Module::new(
                BlockParentLink(parent_module.downgrade(), block_id),
                None,
                AnonymousModuleKind,
                false,
                false));
            parent_module.anonymous_children.borrow_mut()
                         .insert(block_id, new_module.clone());
            ModuleReducedGraphParent(new_module)
        } else {
            parent
        }
    }

    fn handle_external_def(&mut self,
                           def: Def,
                           vis: Visibility,
                           child_name_bindings: &NameBindings,
                           final_ident: &str,
                           ident: Ident,
                           new_parent: ReducedGraphParent) {
        debug!("(building reduced graph for \
                external crate) building external def, priv {:?}",
               vis);
        let is_public = vis == ast::Public;
        let is_exported = is_public && match new_parent {
            ModuleReducedGraphParent(ref module) => {
                match module.def_id.get() {
                    None => true,
                    Some(did) => self.external_exports.contains(&did)
                }
            }
        };
        if is_exported {
            self.external_exports.insert(def_id_of_def(def));
        }
        match def {
          DefMod(def_id) | DefForeignMod(def_id) | DefStruct(def_id) |
          DefTy(def_id) => {
            let type_def = child_name_bindings.type_def.borrow().clone();
            match type_def {
              Some(TypeNsDef { module_def: Some(module_def), .. }) => {
                debug!("(building reduced graph for external crate) \
                        already created module");
                module_def.def_id.set(Some(def_id));
              }
              Some(_) | None => {
                debug!("(building reduced graph for \
                        external crate) building module \
                        {}", final_ident);
                let parent_link = self.get_parent_link(new_parent.clone(), ident);

                child_name_bindings.define_module(parent_link,
                                                  Some(def_id),
                                                  NormalModuleKind,
                                                  true,
                                                  is_public,
                                                  DUMMY_SP);
              }
            }
          }
          _ => {}
        }

        match def {
          DefMod(_) | DefForeignMod(_) => {}
          DefVariant(enum_did, variant_id, is_struct) => {
            debug!("(building reduced graph for external crate) building \
                    variant {}",
                   final_ident);
            // If this variant is public, then it was publicly reexported,
            // otherwise we need to inherit the visibility of the enum
            // definition.
            let is_exported = is_public ||
                              self.external_exports.contains(&enum_did);
            if is_struct {
                child_name_bindings.define_type(def, DUMMY_SP, is_exported);
                // Not adding fields for variants as they are not accessed with a self receiver
                self.structs.insert(variant_id, Vec::new());
            } else {
                child_name_bindings.define_value(def, DUMMY_SP, is_exported);
            }
          }
          DefFn(..) | DefStaticMethod(..) | DefStatic(..) => {
            debug!("(building reduced graph for external \
                    crate) building value (fn/static) {}", final_ident);
            child_name_bindings.define_value(def, DUMMY_SP, is_public);
          }
          DefTrait(def_id) => {
              debug!("(building reduced graph for external \
                      crate) building type {}", final_ident);

              // If this is a trait, add all the method names
              // to the trait info.

              let method_def_ids =
                csearch::get_trait_method_def_ids(&self.session.cstore, def_id);
              for &method_def_id in method_def_ids.iter() {
                  let (method_name, explicit_self) =
                      csearch::get_method_name_and_explicit_self(&self.session.cstore,
                                                                 method_def_id);

                  debug!("(building reduced graph for \
                          external crate) ... adding \
                          trait method '{}'",
                         token::get_ident(method_name));

                  self.method_map.borrow_mut().insert((method_name.name, def_id), explicit_self);

                  if is_exported {
                      self.external_exports.insert(method_def_id);
                  }
              }

              child_name_bindings.define_type(def, DUMMY_SP, is_public);

              // Define a module if necessary.
              let parent_link = self.get_parent_link(new_parent, ident);
              child_name_bindings.set_module_kind(parent_link,
                                                  Some(def_id),
                                                  TraitModuleKind,
                                                  true,
                                                  is_public,
                                                  DUMMY_SP)
          }
          DefTy(_) => {
              debug!("(building reduced graph for external \
                      crate) building type {}", final_ident);

              child_name_bindings.define_type(def, DUMMY_SP, is_public);
          }
          DefStruct(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type and value for {}",
                   final_ident);
            child_name_bindings.define_type(def, DUMMY_SP, is_public);
            let fields = csearch::get_struct_fields(&self.session.cstore, def_id).iter().map(|f| {
                f.name
            }).collect::<Vec<_>>();

            if fields.len() == 0 {
                child_name_bindings.define_value(def, DUMMY_SP, is_public);
            }

            // Record the def ID and fields of this struct.
            self.structs.insert(def_id, fields);
          }
          DefMethod(..) => {
              debug!("(building reduced graph for external crate) \
                      ignoring {:?}", def);
              // Ignored; handled elsewhere.
          }
          DefArg(..) | DefLocal(..) | DefPrimTy(..) |
          DefTyParam(..) | DefBinding(..) |
          DefUse(..) | DefUpvar(..) | DefRegion(..) |
          DefTyParamBinder(..) | DefLabel(..) | DefSelfTy(..) => {
            fail!("didn't expect `{:?}`", def);
          }
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self,
                                                  root: Rc<Module>,
                                                  def_like: DefLike,
                                                  ident: Ident,
                                                  visibility: Visibility) {
        match def_like {
            DlDef(def) => {
                // Add the new child item, if necessary.
                match def {
                    DefForeignMod(def_id) => {
                        // Foreign modules have no names. Recur and populate
                        // eagerly.
                        csearch::each_child_of_item(&self.session.cstore,
                                                    def_id,
                                                    |def_like,
                                                     child_ident,
                                                     vis| {
                            self.build_reduced_graph_for_external_crate_def(
                                root.clone(),
                                def_like,
                                child_ident,
                                vis)
                        });
                    }
                    _ => {
                        let child_name_bindings =
                            self.add_child(ident,
                                           ModuleReducedGraphParent(root.clone()),
                                           OverwriteDuplicates,
                                           DUMMY_SP);

                        self.handle_external_def(def,
                                                 visibility,
                                                 &*child_name_bindings,
                                                 token::get_ident(ident).get(),
                                                 ident,
                                                 ModuleReducedGraphParent(root));
                    }
                }
            }
            DlImpl(def) => {
                // We only process static methods of impls here.
                match csearch::get_type_name_if_impl(&self.session.cstore, def) {
                    None => {}
                    Some(final_ident) => {
                        let static_methods_opt =
                            csearch::get_static_methods_if_impl(&self.session.cstore, def);
                        match static_methods_opt {
                            Some(ref static_methods) if
                                static_methods.len() >= 1 => {
                                debug!("(building reduced graph for \
                                        external crate) processing \
                                        static methods for type name {}",
                                        token::get_ident(final_ident));

                                let child_name_bindings =
                                    self.add_child(
                                        final_ident,
                                        ModuleReducedGraphParent(root.clone()),
                                        OverwriteDuplicates,
                                        DUMMY_SP);

                                // Process the static methods. First,
                                // create the module.
                                let type_module;
                                let type_def = child_name_bindings.type_def.borrow().clone();
                                match type_def {
                                    Some(TypeNsDef {
                                        module_def: Some(module_def),
                                        ..
                                    }) => {
                                        // We already have a module. This
                                        // is OK.
                                        type_module = module_def;

                                        // Mark it as an impl module if
                                        // necessary.
                                        type_module.kind.set(ImplModuleKind);
                                    }
                                    Some(_) | None => {
                                        let parent_link =
                                            self.get_parent_link(ModuleReducedGraphParent(root),
                                                                 final_ident);
                                        child_name_bindings.define_module(
                                            parent_link,
                                            Some(def),
                                            ImplModuleKind,
                                            true,
                                            true,
                                            DUMMY_SP);
                                        type_module =
                                            child_name_bindings.
                                                get_module();
                                    }
                                }

                                // Add each static method to the module.
                                let new_parent =
                                    ModuleReducedGraphParent(type_module);
                                for static_method_info in
                                        static_methods.iter() {
                                    let ident = static_method_info.ident;
                                    debug!("(building reduced graph for \
                                             external crate) creating \
                                             static method '{}'",
                                           token::get_ident(ident));

                                    let method_name_bindings =
                                        self.add_child(ident,
                                                       new_parent.clone(),
                                                       OverwriteDuplicates,
                                                       DUMMY_SP);
                                    let def = DefFn(
                                        static_method_info.def_id,
                                        static_method_info.fn_style);

                                    method_name_bindings.define_value(
                                        def, DUMMY_SP,
                                        visibility == ast::Public);
                                }
                            }

                            // Otherwise, do nothing.
                            Some(_) | None => {}
                        }
                    }
                }
            }
            DlField => {
                debug!("(building reduced graph for external crate) \
                        ignoring field");
            }
        }
    }

    /// Builds the reduced graph rooted at the given external module.
    fn populate_external_module(&mut self, module: Rc<Module>) {
        debug!("(populating external module) attempting to populate {}",
               self.module_to_str(&*module));

        let def_id = match module.def_id.get() {
            None => {
                debug!("(populating external module) ... no def ID!");
                return
            }
            Some(def_id) => def_id,
        };

        csearch::each_child_of_item(&self.session.cstore,
                                    def_id,
                                    |def_like, child_ident, visibility| {
            debug!("(populating external module) ... found ident: {}",
                   token::get_ident(child_ident));
            self.build_reduced_graph_for_external_crate_def(module.clone(),
                                                            def_like,
                                                            child_ident,
                                                            visibility)
        });
        module.populated.set(true)
    }

    /// Ensures that the reduced graph rooted at the given external module
    /// is built, building it if it is not.
    fn populate_module_if_necessary(&mut self, module: &Rc<Module>) {
        if !module.populated.get() {
            self.populate_external_module(module.clone())
        }
        assert!(module.populated.get())
    }

    /// Builds the reduced graph rooted at the 'use' directive for an external
    /// crate.
    fn build_reduced_graph_for_external_crate(&mut self, root: Rc<Module>) {
        csearch::each_top_level_item_of_crate(&self.session.cstore,
                                              root.def_id
                                                  .get()
                                                  .unwrap()
                                                  .krate,
                                              |def_like, ident, visibility| {
            self.build_reduced_graph_for_external_crate_def(root.clone(),
                                                            def_like,
                                                            ident,
                                                            visibility)
        });
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(&mut self,
                              module_: &Module,
                              module_path: Vec<Ident> ,
                              subclass: ImportDirectiveSubclass,
                              span: Span,
                              id: NodeId,
                              is_public: bool) {
        module_.imports.borrow_mut().push(ImportDirective::new(module_path,
                                                               subclass,
                                                               span, id,
                                                               is_public));
        self.unresolved_imports += 1;
        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match subclass {
            SingleImport(target, _) => {
                debug!("(building import directive) building import \
                        directive: {}::{}",
                       self.idents_to_str(module_.imports.borrow().last().unwrap()
                                                 .module_path.as_slice()),
                       token::get_ident(target));

                let mut import_resolutions = module_.import_resolutions
                                                    .borrow_mut();
                match import_resolutions.find_mut(&target.name) {
                    Some(resolution) => {
                        debug!("(building import directive) bumping \
                                reference");
                        resolution.outstanding_references += 1;

                        // the source of this name is different now
                        resolution.type_id = id;
                        resolution.value_id = id;
                        resolution.is_public = is_public;
                        return;
                    }
                    None => {}
                }
                debug!("(building import directive) creating new");
                let mut resolution = ImportResolution::new(id, is_public);
                resolution.outstanding_references = 1;
                import_resolutions.insert(target.name, resolution);
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.glob_count.set(module_.glob_count.get() + 1);
            }
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
               self.module_to_str(&*module_));
        self.resolve_imports_for_module(module_.clone());

        self.populate_module_if_necessary(&module_);
        for (_, child_node) in module_.children.borrow().iter() {
            match child_node.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.resolve_imports_for_module_subtree(child_module);
                }
            }
        }

        for (_, child_module) in module_.anonymous_children.borrow().iter() {
            self.resolve_imports_for_module_subtree(child_module.clone());
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(&mut self, module: Rc<Module>) {
        if module.all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for \
                   {}",
                   self.module_to_str(&*module));
            return;
        }

        let mut imports = module.imports.borrow_mut();
        let import_count = imports.len();
        while module.resolved_import_count.get() < import_count {
            let import_index = module.resolved_import_count.get();
            let import_directive = imports.get(import_index);
            match self.resolve_import_for_module(module.clone(),
                                                 import_directive) {
                Failed => {
                    // We presumably emitted an error. Continue.
                    let msg = format!("failed to resolve import `{}`",
                                   self.import_path_to_str(
                                       import_directive.module_path
                                                       .as_slice(),
                                       import_directive.subclass));
                    self.resolve_error(import_directive.span, msg);
                }
                Indeterminate => {
                    // Bail out. We'll come around next time.
                    break;
                }
                Success(()) => {
                    // Good. Continue.
                }
            }

            module.resolved_import_count
                  .set(module.resolved_import_count.get() + 1);
        }
    }

    fn idents_to_str(&self, idents: &[Ident]) -> StrBuf {
        let mut first = true;
        let mut result = StrBuf::new();
        for ident in idents.iter() {
            if first {
                first = false
            } else {
                result.push_str("::")
            }
            result.push_str(token::get_ident(*ident).get());
        };
        result
    }

    fn path_idents_to_str(&self, path: &Path) -> StrBuf {
        let identifiers: Vec<ast::Ident> = path.segments
                                             .iter()
                                             .map(|seg| seg.identifier)
                                             .collect();
        self.idents_to_str(identifiers.as_slice())
    }

    fn import_directive_subclass_to_str(&mut self,
                                        subclass: ImportDirectiveSubclass)
                                        -> StrBuf {
        match subclass {
            SingleImport(_, source) => {
                token::get_ident(source).get().to_strbuf()
            }
            GlobImport => "*".to_strbuf()
        }
    }

    fn import_path_to_str(&mut self,
                          idents: &[Ident],
                          subclass: ImportDirectiveSubclass)
                          -> StrBuf {
        if idents.is_empty() {
            self.import_directive_subclass_to_str(subclass)
        } else {
            (format!("{}::{}",
                     self.idents_to_str(idents),
                     self.import_directive_subclass_to_str(
                         subclass))).to_strbuf()
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
        let mut resolution_result = Failed;
        let module_path = &import_directive.module_path;

        debug!("(resolving import for module) resolving import `{}::...` in \
                `{}`",
               self.idents_to_str(module_path.as_slice()),
               self.module_to_str(&*module_));

        // First, resolve the module path for the directive, if necessary.
        let container = if module_path.len() == 0 {
            // Use the crate root.
            Some((self.graph_root.get_module(), LastMod(AllPublic)))
        } else {
            match self.resolve_module_path(module_.clone(),
                                           module_path.as_slice(),
                                           DontUseLexicalScope,
                                           import_directive.span,
                                           ImportSearch) {

                Failed => None,
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
                                                     import_directive.id,
                                                     import_directive.is_public,
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
                is_public: false,
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
                             target: Ident,
                             source: Ident,
                             directive: &ImportDirective,
                             lp: LastPrivate)
                                 -> ResolveResult<()> {
        debug!("(resolving single import) resolving `{}` = `{}::{}` from \
                `{}` id {}, last private {:?}",
               token::get_ident(target),
               self.module_to_str(&*containing_module),
               token::get_ident(source),
               self.module_to_str(module_),
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
        self.populate_module_if_necessary(&containing_module);

        match containing_module.children.borrow().find(&source.name) {
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
                match containing_module.import_resolutions.borrow().find(&source.name) {
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
                                       namespace: Namespace)
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
                                Some(Target {target_module, bindings}) => {
                                    debug!("(resolving single import) found \
                                            import in ns {:?}", namespace);
                                    let id = import_resolution.id(namespace);
                                    this.used_imports.insert((id, namespace));
                                    return BoundResult(target_module, bindings);
                                }
                            }
                        }

                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.
                        if value_result.is_unknown() {
                            value_result = get_binding(self, import_resolution,
                                                       ValueNS);
                            value_used_reexport = import_resolution.is_public;
                        }
                        if type_result.is_unknown() {
                            type_result = get_binding(self, import_resolution,
                                                      TypeNS);
                            type_used_reexport = import_resolution.is_public;
                        }

                    }
                    Some(_) => {
                        // The import is unresolved. Bail out.
                        debug!("(resolving single import) unresolved import; \
                                bailing out");
                        return Indeterminate;
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
                                       .find_copy(&source.name) {
                    None => {} // Continue.
                    Some(module) => {
                        debug!("(resolving single import) found external \
                                module");
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
        let import_resolution = import_resolutions.get_mut(&target.name);

        match value_result {
            BoundResult(ref target_module, ref name_bindings) => {
                debug!("(resolving single import) found value target");
                import_resolution.value_target = Some(Target::new(target_module.clone(),
                                                                  name_bindings.clone()));
                import_resolution.value_id = directive.id;
                import_resolution.is_public = directive.is_public;
                value_used_public = name_bindings.defined_in_public_namespace(ValueNS);
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!("value result should be known at this point");
            }
        }
        match type_result {
            BoundResult(ref target_module, ref name_bindings) => {
                debug!("(resolving single import) found type target: {:?}",
                       { name_bindings.type_def.borrow().clone().unwrap().type_def });
                import_resolution.type_target =
                    Some(Target::new(target_module.clone(), name_bindings.clone()));
                import_resolution.type_id = directive.id;
                import_resolution.is_public = directive.is_public;
                type_used_public = name_bindings.defined_in_public_namespace(TypeNS);
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!("type result should be known at this point");
            }
        }

        if value_result.is_unbound() && type_result.is_unbound() {
            let msg = format!("unresolved import: there is no \
                               `{}` in `{}`",
                              token::get_ident(source),
                              self.module_to_str(&*containing_module));
            self.resolve_error(directive.span, msg);
            return Failed;
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
                let did = def_id_of_def(def);
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
                let did = def_id_of_def(def);
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
    // that exports nothing is valid).
    fn resolve_glob_import(&mut self,
                           module_: &Module,
                           containing_module: Rc<Module>,
                           id: NodeId,
                           is_public: bool,
                           lp: LastPrivate)
                           -> ResolveResult<()> {
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
        let import_resolutions = containing_module.import_resolutions
                                                  .borrow();
        for (ident, target_import_resolution) in import_resolutions.iter() {
            debug!("(resolving glob import) writing module resolution \
                    {:?} into `{}`",
                   target_import_resolution.type_target.is_none(),
                   self.module_to_str(module_));

            if !target_import_resolution.is_public {
                debug!("(resolving glob import) nevermind, just kidding");
                continue
            }

            // Here we merge two import resolutions.
            let mut import_resolutions = module_.import_resolutions.borrow_mut();
            match import_resolutions.find_mut(ident) {
                Some(dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match target_import_resolution.value_target {
                        None => {
                            // Continue.
                        }
                        Some(ref value_target) => {
                            dest_import_resolution.value_target =
                                Some(value_target.clone());
                        }
                    }
                    match target_import_resolution.type_target {
                        None => {
                            // Continue.
                        }
                        Some(ref type_target) => {
                            dest_import_resolution.type_target =
                                Some(type_target.clone());
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
        self.populate_module_if_necessary(&containing_module);

        for (&name, name_bindings) in containing_module.children
                                                       .borrow().iter() {
            self.merge_import_resolution(module_, containing_module.clone(),
                                         id, is_public,
                                         name, name_bindings.clone());
        }

        // Add external module children from the containing module.
        for (&name, module) in containing_module.external_module_children
                                                .borrow().iter() {
            let name_bindings =
                Rc::new(Resolver::create_name_bindings_from_module(module.clone()));
            self.merge_import_resolution(module_, containing_module.clone(),
                                         id, is_public,
                                         name, name_bindings);
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
                               id: NodeId,
                               is_public: bool,
                               name: Name,
                               name_bindings: Rc<NameBindings>) {
        let mut import_resolutions = module_.import_resolutions.borrow_mut();
        let dest_import_resolution = import_resolutions.find_or_insert_with(name, |_| {
            // Create a new import resolution from this child.
            ImportResolution::new(id, is_public)
        });

        debug!("(resolving glob import) writing resolution `{}` in `{}` \
               to `{}`",
               token::get_name(name).get().to_str(),
               self.module_to_str(&*containing_module),
               self.module_to_str(module_));

        // Merge the child item into the import resolution.
        if name_bindings.defined_in_public_namespace(ValueNS) {
            debug!("(resolving glob import) ... for value target");
            dest_import_resolution.value_target =
                Some(Target::new(containing_module.clone(), name_bindings.clone()));
            dest_import_resolution.value_id = id;
        }
        if name_bindings.defined_in_public_namespace(TypeNS) {
            debug!("(resolving glob import) ... for type target");
            dest_import_resolution.type_target =
                Some(Target::new(containing_module, name_bindings.clone()));
            dest_import_resolution.type_id = id;
        }
        dest_import_resolution.is_public = is_public;
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(&mut self,
                                     module_: Rc<Module>,
                                     module_path: &[Ident],
                                     index: uint,
                                     span: Span,
                                     name_search_type: NameSearchType,
                                     lp: LastPrivate)
                                -> ResolveResult<(Rc<Module>, LastPrivate)> {
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
                                              name.name,
                                              TypeNS,
                                              name_search_type,
                                              false) {
                Failed => {
                    let segment_name = token::get_ident(name);
                    let module_name = self.module_to_str(&*search_module);
                    if "???" == module_name.as_slice() {
                        let span = Span {
                            lo: span.lo,
                            hi: span.lo + Pos::from_uint(segment_name.get().len()),
                            expn_info: span.expn_info,
                        };
                        self.resolve_error(span,
                                              format!("unresolved import. maybe \
                                                    a missing `extern crate \
                                                    {}`?",
                                                    segment_name));
                        return Failed;
                    }
                    self.resolve_error(span, format!("unresolved import: could not find `{}` in \
                                                     `{}`.", segment_name, module_name));
                    return Failed;
                }
                Indeterminate => {
                    debug!("(resolving module path for import) module \
                            resolution is indeterminate: {}",
                            token::get_ident(name));
                    return Indeterminate;
                }
                Success((target, used_proxy)) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    match *target.bindings.type_def.borrow() {
                        Some(ref type_def) => {
                            match type_def.module_def {
                                None => {
                                    // Not a module.
                                    self.resolve_error(span, format!("not a module `{}`",
                                                                 token::get_ident(name)));
                                    return Failed;
                                }
                                Some(ref module_def) => {
                                    // If we're doing the search for an
                                    // import, do not allow traits and impls
                                    // to be selected.
                                    match (name_search_type,
                                           module_def.kind.get()) {
                                        (ImportSearch, TraitModuleKind) |
                                        (ImportSearch, ImplModuleKind) => {
                                            self.resolve_error(
                                                span,
                                                "cannot import from a trait \
                                                 or type implementation");
                                            return Failed;
                                        }
                                        (_, _) => {
                                            search_module = module_def.clone();

                                            // Keep track of the closest
                                            // private module used when
                                            // resolving this import chain.
                                            if !used_proxy &&
                                               !search_module.is_public {
                                                match search_module.def_id
                                                                   .get() {
                                                    Some(did) => {
                                                        closest_private =
                                                            LastMod(DependsOn(did));
                                                    }
                                                    None => {}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            // There are no type bindings at all.
                            self.resolve_error(span,
                                                  format!("not a module `{}`",
                                                       token::get_ident(name)));
                            return Failed;
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
                           module_path: &[Ident],
                           use_lexical_scope: UseLexicalScopeFlag,
                           span: Span,
                           name_search_type: NameSearchType)
                               -> ResolveResult<(Rc<Module>, LastPrivate)> {
        let module_path_len = module_path.len();
        assert!(module_path_len > 0);

        debug!("(resolving module path for import) processing `{}` rooted at \
               `{}`",
               self.idents_to_str(module_path),
               self.module_to_str(&*module_));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_.clone(),
                                                              module_path);

        let search_module;
        let start_index;
        let last_private;
        match module_prefix_result {
            Failed => {
                let mpath = self.idents_to_str(module_path);
                match mpath.as_slice().rfind(':') {
                    Some(idx) => {
                        self.resolve_error(span,
                                           format!("unresolved import: could \
                                                    not find `{}` in `{}`",
                                                   // idx +- 1 to account for
                                                   // the colons on either
                                                   // side
                                                   mpath.as_slice()
                                                        .slice_from(idx + 1),
                                                   mpath.as_slice()
                                                        .slice_to(idx - 1)));
                    },
                    None => (),
                };
                return Failed;
            }
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
                        let result = self.resolve_module_in_lexical_scope(
                            module_,
                            module_path[0]);
                        match result {
                            Failed => {
                                self.resolve_error(span, "unresolved name");
                                return Failed;
                            }
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
                                     name: Ident,
                                     namespace: Namespace)
                                    -> ResolveResult<(Target, bool)> {
        debug!("(resolving item in lexical scope) resolving `{}` in \
                namespace {:?} in `{}`",
               token::get_ident(name),
               namespace,
               self.module_to_str(&*module_));

        // The current module node is handled specially. First, check for
        // its immediate children.
        self.populate_module_if_necessary(&module_);

        match module_.children.borrow().find(&name.name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                debug!("top name bindings succeeded");
                return Success((Target::new(module_.clone(), name_bindings.clone()),
                               false));
            }
            Some(_) | None => { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.
        match module_.import_resolutions.borrow().find(&name.name) {
            None => {
                // Not found; continue.
            }
            Some(import_resolution) => {
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
                        self.used_imports.insert((import_resolution.id(namespace), namespace));
                        return Success((target, false));
                    }
                }
            }
        }

        // Search for external modules.
        if namespace == TypeNS {
            match module_.external_module_children.borrow().find_copy(&name.name) {
                None => {}
                Some(module) => {
                    let name_bindings =
                        Rc::new(Resolver::create_name_bindings_from_module(module));
                    debug!("lower name bindings succeeded");
                    return Success((Target::new(module_, name_bindings), false));
                }
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
                    return Failed;
                }
                ModuleParentLink(parent_module_node, _) => {
                    match search_module.kind.get() {
                        NormalModuleKind => {
                            // We stop the search here.
                            debug!("(resolving item in lexical \
                                    scope) unresolved module: not \
                                    searching through module \
                                    parents");
                            return Failed;
                        }
                        ExternModuleKind |
                        TraitModuleKind |
                        ImplModuleKind |
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
                                              name.name,
                                              namespace,
                                              PathSearch,
                                              true) {
                Failed => {
                    // Continue up the search chain.
                }
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
                                       name: Ident)
                                -> ResolveResult<Rc<Module>> {
        // If this module is an anonymous module, resolve the item in the
        // lexical scope. Otherwise, resolve the item from the crate root.
        let resolve_result = self.resolve_item_in_lexical_scope(
            module_, name, TypeNS);
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
                                return Failed;
                            }
                            Some(ref module_def) => {
                                return Success(module_def.clone());
                            }
                        }
                    }
                    None => {
                        debug!("!!! (resolving module in lexical scope) module
                                wasn't actually a module!");
                        return Failed;
                    }
                }
            }
            Indeterminate => {
                debug!("(resolving module in lexical scope) indeterminate; \
                        bailing");
                return Indeterminate;
            }
            Failed => {
                debug!("(resolving module in lexical scope) failed to \
                        resolve");
                return Failed;
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
                        ExternModuleKind |
                        TraitModuleKind |
                        ImplModuleKind |
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
            ExternModuleKind |
            TraitModuleKind |
            ImplModuleKind |
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
                             module_path: &[Ident])
                                 -> ResolveResult<ModulePrefixResult> {
        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut containing_module;
        let mut i;
        let first_module_path_string = token::get_ident(module_path[0]);
        if "self" == first_module_path_string.get() {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 1;
        } else if "super" == first_module_path_string.get() {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 0;  // We'll handle `super` below.
        } else {
            return Success(NoPrefixFound);
        }

        // Now loop through all the `super`s we find.
        while i < module_path.len() {
            let string = token::get_ident(module_path[i]);
            if "super" != string.get() {
                break
            }
            debug!("(resolving module prefix) resolving `super` at {}",
                   self.module_to_str(&*containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => return Failed,
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at {}",
               self.module_to_str(&*containing_module));

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
               token::get_name(name).get(),
               self.module_to_str(&*module_));

        // First, check the direct children of the module.
        self.populate_module_if_necessary(&module_);

        match module_.children.borrow().find(&name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                debug!("(resolving name in module) found node as child");
                return Success((Target::new(module_.clone(), name_bindings.clone()),
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
        match module_.import_resolutions.borrow().find(&name) {
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
                        self.used_imports.insert((import_resolution.id(namespace), namespace));
                        return Success((target, true));
                    }
                }
            }
            Some(..) | None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            match module_.external_module_children.borrow().find_copy(&name) {
                None => {}
                Some(module) => {
                    let name_bindings =
                        Rc::new(Resolver::create_name_bindings_from_module(module));
                    return Success((Target::new(module_, name_bindings), false));
                }
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve `{}`",
               token::get_name(name).get());
        return Failed;
    }

    fn report_unresolved_imports(&mut self, module_: Rc<Module>) {
        let index = module_.resolved_import_count.get();
        let imports = module_.imports.borrow();
        let import_count = imports.len();
        if index != import_count {
            let sn = self.session
                         .codemap()
                         .span_to_snippet(imports.get(index).span)
                         .unwrap();
            if sn.as_slice().contains("::") {
                self.resolve_error(imports.get(index).span,
                                   "unresolved import");
            } else {
                let err = format!("unresolved import (maybe you meant `{}::*`?)",
                                  sn.as_slice().slice(0, sn.len()));
                self.resolve_error(imports.get(index).span, err);
            }
        }

        // Descend into children and anonymous children.
        self.populate_module_if_necessary(&module_);

        for (_, child_node) in module_.children.borrow().iter() {
            match child_node.get_module_if_available() {
                None => {
                    // Continue.
                }
                Some(child_module) => {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for (_, module_) in module_.anonymous_children.borrow().iter() {
            self.report_unresolved_imports(module_.clone());
        }
    }

    // Export recording
    //
    // This pass simply determines what all "export" keywords refer to and
    // writes the results into the export map.
    //
    // FIXME #4953 This pass will be removed once exports change to per-item.
    // Then this operation can simply be performed as part of item (or import)
    // processing.

    fn record_exports(&mut self) {
        let root_module = self.graph_root.get_module();
        self.record_exports_for_module_subtree(root_module);
    }

    fn record_exports_for_module_subtree(&mut self,
                                             module_: Rc<Module>) {
        // If this isn't a local krate, then bail out. We don't need to record
        // exports for nonlocal crates.

        match module_.def_id.get() {
            Some(def_id) if def_id.krate == LOCAL_CRATE => {
                // OK. Continue.
                debug!("(recording exports for module subtree) recording \
                        exports for local module `{}`",
                       self.module_to_str(&*module_));
            }
            None => {
                // Record exports for the root module.
                debug!("(recording exports for module subtree) recording \
                        exports for root module `{}`",
                       self.module_to_str(&*module_));
            }
            Some(_) => {
                // Bail out.
                debug!("(recording exports for module subtree) not recording \
                        exports for `{}`",
                       self.module_to_str(&*module_));
                return;
            }
        }

        self.record_exports_for_module(&*module_);
        self.populate_module_if_necessary(&module_);

        for (_, child_name_bindings) in module_.children.borrow().iter() {
            match child_name_bindings.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.record_exports_for_module_subtree(child_module);
                }
            }
        }

        for (_, child_module) in module_.anonymous_children.borrow().iter() {
            self.record_exports_for_module_subtree(child_module.clone());
        }
    }

    fn record_exports_for_module(&mut self, module_: &Module) {
        let mut exports2 = Vec::new();

        self.add_exports_for_module(&mut exports2, module_);
        match module_.def_id.get() {
            Some(def_id) => {
                self.export_map2.borrow_mut().insert(def_id.node, exports2);
                debug!("(computing exports) writing exports for {} (some)",
                       def_id.node);
            }
            None => {}
        }
    }

    fn add_exports_of_namebindings(&mut self,
                                   exports2: &mut Vec<Export2> ,
                                   name: Name,
                                   namebindings: &NameBindings,
                                   ns: Namespace) {
        match namebindings.def_for_namespace(ns) {
            Some(d) => {
                let name = token::get_name(name);
                debug!("(computing exports) YES: export '{}' => {:?}",
                       name, def_id_of_def(d));
                exports2.push(Export2 {
                    name: name.get().to_strbuf(),
                    def_id: def_id_of_def(d)
                });
            }
            d_opt => {
                debug!("(computing exports) NO: {:?}", d_opt);
            }
        }
    }

    fn add_exports_for_module(&mut self,
                              exports2: &mut Vec<Export2> ,
                              module_: &Module) {
        for (name, importresolution) in module_.import_resolutions.borrow().iter() {
            if !importresolution.is_public {
                continue
            }
            let xs = [TypeNS, ValueNS];
            for &ns in xs.iter() {
                match importresolution.target_for_namespace(ns) {
                    Some(target) => {
                        debug!("(computing exports) maybe export '{}'",
                               token::get_name(*name));
                        self.add_exports_of_namebindings(exports2,
                                                         *name,
                                                         &*target.bindings,
                                                         ns)
                    }
                    _ => ()
                }
            }
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

    fn with_scope(&mut self, name: Option<Ident>, f: |&mut Resolver|) {
        let orig_module = self.current_module.clone();

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                self.populate_module_if_necessary(&orig_module);

                match orig_module.children.borrow().find(&name.name) {
                    None => {
                        debug!("!!! (with scope) didn't find `{}` in `{}`",
                               token::get_ident(name),
                               self.module_to_str(&*orig_module));
                    }
                    Some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            None => {
                                debug!("!!! (with scope) didn't find module \
                                        for `{}` in `{}`",
                                       token::get_ident(name),
                                       self.module_to_str(&*orig_module));
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

    /// Wraps the given definition in the appropriate number of `def_upvar`
    /// wrappers.
    fn upvarify(&self,
                ribs: &[Rib],
                rib_index: uint,
                def_like: DefLike,
                span: Span)
                -> Option<DefLike> {
        let mut def;
        let is_ty_param;

        match def_like {
            DlDef(d @ DefLocal(..)) | DlDef(d @ DefUpvar(..)) |
            DlDef(d @ DefArg(..)) | DlDef(d @ DefBinding(..)) => {
                def = d;
                is_ty_param = false;
            }
            DlDef(d @ DefTyParam(..)) => {
                def = d;
                is_ty_param = true;
            }
            _ => {
                return Some(def_like);
            }
        }

        let mut rib_index = rib_index + 1;
        while rib_index < ribs.len() {
            match ribs[rib_index].kind {
                NormalRibKind => {
                    // Nothing to do. Continue.
                }
                FunctionRibKind(function_id, body_id) => {
                    if !is_ty_param {
                        def = DefUpvar(def_id_of_def(def).node,
                                        @def,
                                        function_id,
                                        body_id);
                    }
                }
                MethodRibKind(item_id, _) => {
                  // If the def is a ty param, and came from the parent
                  // item, it's ok
                  match def {
                    DefTyParam(did, _) if {
                        self.def_map.borrow().find(&did.node).map(|x| *x)
                            == Some(DefTyParamBinder(item_id))
                    } => {
                      // ok
                    }
                    _ => {
                    if !is_ty_param {
                        // This was an attempt to access an upvar inside a
                        // named function item. This is not allowed, so we
                        // report an error.

                        self.resolve_error(
                            span,
                            "can't capture dynamic environment in a fn item; \
                            use the || { ... } closure form instead");
                    } else {
                        // This was an attempt to use a type parameter outside
                        // its scope.

                        self.resolve_error(span,
                                              "can't use type parameters from \
                                              outer function; try using a local \
                                              type parameter instead");
                    }

                    return None;
                    }
                  }
                }
                OpaqueFunctionRibKind => {
                    if !is_ty_param {
                        // This was an attempt to access an upvar inside a
                        // named function item. This is not allowed, so we
                        // report an error.

                        self.resolve_error(
                            span,
                            "can't capture dynamic environment in a fn item; \
                            use the || { ... } closure form instead");
                    } else {
                        // This was an attempt to use a type parameter outside
                        // its scope.

                        self.resolve_error(span,
                                              "can't use type parameters from \
                                              outer function; try using a local \
                                              type parameter instead");
                    }

                    return None;
                }
                ConstantItemRibKind => {
                    if is_ty_param {
                        // see #9186
                        self.resolve_error(span,
                                              "cannot use an outer type \
                                               parameter in this context");
                    } else {
                        // Still doesn't deal with upvars
                        self.resolve_error(span,
                                              "attempt to use a non-constant \
                                               value in a constant");
                    }

                }
            }

            rib_index += 1;
        }

        return Some(DlDef(def));
    }

    fn search_ribs(&self,
                   ribs: &[Rib],
                   name: Name,
                   span: Span)
                   -> Option<DefLike> {
        // FIXME #4950: This should not use a while loop.
        // FIXME #4950: Try caching?

        let mut i = ribs.len();
        while i != 0 {
            i -= 1;
            let binding_opt = ribs[i].bindings.borrow().find_copy(&name);
            match binding_opt {
                Some(def_like) => {
                    return self.upvarify(ribs, i, def_like, span);
                }
                None => {
                    // Continue.
                }
            }
        }

        return None;
    }

    fn resolve_crate(&mut self, krate: &ast::Crate) {
        debug!("(resolving crate) starting");

        visit::walk_crate(self, krate, ());
    }

    fn resolve_item(&mut self, item: &Item) {
        debug!("(resolving item) resolving {}",
               token::get_ident(item.ident));

        match item.node {

            // enum item: resolve all the variants' discrs,
            // then resolve the ty params
            ItemEnum(ref enum_def, ref generics) => {
                for variant in (*enum_def).variants.iter() {
                    for dis_expr in variant.node.disr_expr.iter() {
                        // resolve the discriminator expr
                        // as a constant
                        self.with_constant_rib(|this| {
                            this.resolve_expr(*dis_expr);
                        });
                    }
                }

                // n.b. the discr expr gets visited twice.
                // but maybe it's okay since the first time will signal an
                // error if there is one? -- tjc
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               item.id,
                                                               0,
                                                               NormalRibKind),
                                             |this| {
                    visit::walk_item(this, item, ());
                });
            }

            ItemTy(_, ref generics) => {
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               item.id,
                                                               0,
                                                               NormalRibKind),
                                             |this| {
                    visit::walk_item(this, item, ());
                });
            }

            ItemImpl(ref generics,
                      ref implemented_traits,
                      self_type,
                      ref methods) => {
                self.resolve_implementation(item.id,
                                            generics,
                                            implemented_traits,
                                            self_type,
                                            methods.as_slice());
            }

            ItemTrait(ref generics, _, ref traits, ref methods) => {
                // Create a new rib for the self type.
                let self_type_rib = Rib::new(NormalRibKind);
                // plain insert (no renaming)
                let name = self.type_self_ident.name;
                self_type_rib.bindings.borrow_mut()
                             .insert(name, DlDef(DefSelfTy(item.id)));
                self.type_ribs.borrow_mut().push(self_type_rib);

                // Create a new rib for the trait-wide type parameters.
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               item.id,
                                                               0,
                                                               NormalRibKind),
                                             |this| {
                    this.resolve_type_parameters(&generics.ty_params);

                    // Resolve derived traits.
                    for trt in traits.iter() {
                        this.resolve_trait_reference(item.id, trt, TraitDerivation);
                    }

                    for method in (*methods).iter() {
                        // Create a new rib for the method-specific type
                        // parameters.
                        //
                        // FIXME #4951: Do we need a node ID here?

                        match *method {
                          ast::Required(ref ty_m) => {
                            this.with_type_parameter_rib
                                (HasTypeParameters(&ty_m.generics,
                                                   item.id,
                                                   generics.ty_params.len(),
                                        MethodRibKind(item.id, Required)),
                                 |this| {

                                // Resolve the method-specific type
                                // parameters.
                                this.resolve_type_parameters(
                                    &ty_m.generics.ty_params);

                                for argument in ty_m.decl.inputs.iter() {
                                    this.resolve_type(argument.ty);
                                }

                                this.resolve_type(ty_m.decl.output);
                            });
                          }
                          ast::Provided(m) => {
                              this.resolve_method(MethodRibKind(item.id,
                                                     Provided(m.id)),
                                                  m,
                                                  generics.ty_params.len())
                          }
                        }
                    }
                });

                self.type_ribs.borrow_mut().pop();
            }

            ItemStruct(ref struct_def, ref generics) => {
                self.resolve_struct(item.id,
                                    generics,
                                    struct_def.super_struct,
                                    struct_def.fields.as_slice());
            }

            ItemMod(ref module_) => {
                self.with_scope(Some(item.ident), |this| {
                    this.resolve_module(module_, item.span, item.ident,
                                        item.id);
                });
            }

            ItemForeignMod(ref foreign_module) => {
                self.with_scope(Some(item.ident), |this| {
                    for foreign_item in foreign_module.items.iter() {
                        match foreign_item.node {
                            ForeignItemFn(_, ref generics) => {
                                this.with_type_parameter_rib(
                                    HasTypeParameters(
                                        generics, foreign_item.id, 0,
                                        NormalRibKind),
                                    |this| visit::walk_foreign_item(this,
                                                                *foreign_item,
                                                                ()));
                            }
                            ForeignItemStatic(..) => {
                                visit::walk_foreign_item(this,
                                                         *foreign_item,
                                                         ());
                            }
                        }
                    }
                });
            }

            ItemFn(fn_decl, _, _, ref generics, block) => {
                self.resolve_function(OpaqueFunctionRibKind,
                                      Some(fn_decl),
                                      HasTypeParameters
                                        (generics,
                                         item.id,
                                         0,
                                         OpaqueFunctionRibKind),
                                      block);
            }

            ItemStatic(..) => {
                self.with_constant_rib(|this| {
                    visit::walk_item(this, item, ());
                });
            }

           ItemMac(..) => {
                // do nothing, these are just around to be encoded
           }
        }
    }

    fn with_type_parameter_rib(&mut self,
                               type_parameters: TypeParameters,
                               f: |&mut Resolver|) {
        match type_parameters {
            HasTypeParameters(generics, node_id, initial_index,
                              rib_kind) => {

                let function_type_rib = Rib::new(rib_kind);

                for (index, type_parameter) in generics.ty_params.iter().enumerate() {
                    let ident = type_parameter.ident;
                    debug!("with_type_parameter_rib: {} {}", node_id,
                           type_parameter.id);
                    let def_like = DlDef(DefTyParam
                        (local_def(type_parameter.id),
                         index + initial_index));
                    // Associate this type parameter with
                    // the item that bound it
                    self.record_def(type_parameter.id,
                                    (DefTyParamBinder(node_id), LastMod(AllPublic)));
                    // plain insert (no renaming)
                    function_type_rib.bindings.borrow_mut()
                                     .insert(ident.name, def_like);
                }
                self.type_ribs.borrow_mut().push(function_type_rib);
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        match type_parameters {
            HasTypeParameters(..) => { self.type_ribs.borrow_mut().pop(); }
            NoTypeParameters => { }
        }
    }

    fn with_label_rib(&mut self, f: |&mut Resolver|) {
        self.label_ribs.borrow_mut().push(Rib::new(NormalRibKind));
        f(self);
        self.label_ribs.borrow_mut().pop();
    }

    fn with_constant_rib(&mut self, f: |&mut Resolver|) {
        self.value_ribs.borrow_mut().push(Rib::new(ConstantItemRibKind));
        self.type_ribs.borrow_mut().push(Rib::new(ConstantItemRibKind));
        f(self);
        self.type_ribs.borrow_mut().pop();
        self.value_ribs.borrow_mut().pop();
    }

    fn resolve_function(&mut self,
                        rib_kind: RibKind,
                        optional_declaration: Option<P<FnDecl>>,
                        type_parameters: TypeParameters,
                        block: P<Block>) {
        // Create a value rib for the function.
        let function_value_rib = Rib::new(rib_kind);
        self.value_ribs.borrow_mut().push(function_value_rib);

        // Create a label rib for the function.
        let function_label_rib = Rib::new(rib_kind);
        self.label_ribs.borrow_mut().push(function_label_rib);

        // If this function has type parameters, add them now.
        self.with_type_parameter_rib(type_parameters, |this| {
            // Resolve the type parameters.
            match type_parameters {
                NoTypeParameters => {
                    // Continue.
                }
                HasTypeParameters(ref generics, _, _, _) => {
                    this.resolve_type_parameters(&generics.ty_params);
                }
            }

            // Add each argument to the rib.
            match optional_declaration {
                None => {
                    // Nothing to do.
                }
                Some(declaration) => {
                    for argument in declaration.inputs.iter() {
                        this.resolve_pattern(argument.pat,
                                             ArgumentIrrefutableMode,
                                             None);

                        this.resolve_type(argument.ty);

                        debug!("(resolving function) recorded argument");
                    }

                    this.resolve_type(declaration.output);
                }
            }

            // Resolve the function body.
            this.resolve_block(block);

            debug!("(resolving function) leaving function");
        });

        self.label_ribs.borrow_mut().pop();
        self.value_ribs.borrow_mut().pop();
    }

    fn resolve_type_parameters(&mut self,
                                   type_parameters: &OwnedSlice<TyParam>) {
        for type_parameter in type_parameters.iter() {
            for bound in type_parameter.bounds.iter() {
                self.resolve_type_parameter_bound(type_parameter.id, bound);
            }
            match type_parameter.default {
                Some(ty) => self.resolve_type(ty),
                None => {}
            }
        }
    }

    fn resolve_type_parameter_bound(&mut self,
                                        id: NodeId,
                                        type_parameter_bound: &TyParamBound) {
        match *type_parameter_bound {
            TraitTyParamBound(ref tref) => {
                self.resolve_trait_reference(id, tref, TraitBoundingTypeParameter)
            }
            StaticRegionTyParamBound => {}
            OtherRegionTyParamBound(_) => {}
        }
    }

    fn resolve_trait_reference(&mut self,
                                   id: NodeId,
                                   trait_reference: &TraitRef,
                                   reference_type: TraitReferenceType) {
        match self.resolve_path(id, &trait_reference.path, TypeNS, true) {
            None => {
                let path_str = self.path_idents_to_str(&trait_reference.path);
                let usage_str = match reference_type {
                    TraitBoundingTypeParameter => "bound type parameter with",
                    TraitImplementation        => "implement",
                    TraitDerivation            => "derive"
                };

                let msg = format!("attempt to {} a nonexistent trait `{}`", usage_str, path_str);
                self.resolve_error(trait_reference.path.span, msg);
            }
            Some(def) => {
                debug!("(resolving trait) found trait def: {:?}", def);
                self.record_def(trait_reference.ref_id, def);
            }
        }
    }

    fn resolve_struct(&mut self,
                      id: NodeId,
                      generics: &Generics,
                      super_struct: Option<P<Ty>>,
                      fields: &[StructField]) {
        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       id,
                                                       0,
                                                       OpaqueFunctionRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.resolve_type_parameters(&generics.ty_params);

            // Resolve the super struct.
            match super_struct {
                Some(t) => match t.node {
                    TyPath(ref path, None, path_id) => {
                        match this.resolve_path(id, path, TypeNS, true) {
                            Some((DefTy(def_id), lp)) if this.structs.contains_key(&def_id) => {
                                let def = DefStruct(def_id);
                                debug!("(resolving struct) resolved `{}` to type {:?}",
                                       token::get_ident(path.segments
                                                            .last().unwrap()
                                                            .identifier),
                                       def);
                                debug!("(resolving struct) writing resolution for `{}` (id {})",
                                       this.path_idents_to_str(path),
                                       path_id);
                                this.record_def(path_id, (def, lp));
                            }
                            Some((DefStruct(_), _)) => {
                                this.session.span_err(t.span,
                                                      "super-struct is defined \
                                                       in a different crate")
                            },
                            Some(_) => this.session.span_err(t.span,
                                                             "super-struct is not a struct type"),
                            None => this.session.span_err(t.span,
                                                          "super-struct could not be resolved"),
                        }
                    },
                    _ => this.session.span_bug(t.span, "path not mapped to a TyPath")
                },
                None => {}
            }

            // Resolve fields.
            for field in fields.iter() {
                this.resolve_type(field.node.ty);
            }
        });
    }

    // Does this really need to take a RibKind or is it always going
    // to be NormalRibKind?
    fn resolve_method(&mut self,
                      rib_kind: RibKind,
                      method: &Method,
                      outer_type_parameter_count: uint) {
        let method_generics = &method.generics;
        let type_parameters =
            HasTypeParameters(method_generics,
                              method.id,
                              outer_type_parameter_count,
                              rib_kind);

        self.resolve_function(rib_kind, Some(method.decl), type_parameters, method.body);
    }

    fn with_current_self_type<T>(&mut self, self_type: &Ty, f: |&mut Resolver| -> T) -> T {
        // Handle nested impls (inside fn bodies)
        let previous_value = replace(&mut self.current_self_type, Some(self_type.clone()));
        let result = f(self);
        self.current_self_type = previous_value;
        result
    }

    fn with_optional_trait_ref<T>(&mut self, id: NodeId,
                                  opt_trait_ref: &Option<TraitRef>,
                                  f: |&mut Resolver| -> T) -> T {
        let new_val = match *opt_trait_ref {
            Some(ref trait_ref) => {
                self.resolve_trait_reference(id, trait_ref, TraitImplementation);

                match self.def_map.borrow().find(&trait_ref.ref_id) {
                    Some(def) => {
                        let did = def_id_of_def(*def);
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
                                  methods: &[@Method]) {
        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = generics.ty_params.len();
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       id,
                                                       0,
                                                       NormalRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.resolve_type_parameters(&generics.ty_params);

            // Resolve the trait reference, if necessary.
            this.with_optional_trait_ref(id, opt_trait_reference, |this| {
                // Resolve the self type.
                this.resolve_type(self_type);

                this.with_current_self_type(self_type, |this| {
                    for method in methods.iter() {
                        // We also need a new scope for the method-specific type parameters.
                        this.resolve_method(MethodRibKind(id, Provided(method.id)),
                                            *method,
                                            outer_type_parameter_count);
                    }
                });
            });
        });
    }

    fn resolve_module(&mut self, module: &Mod, _span: Span,
                      _name: Ident, id: NodeId) {
        // Write the implementations in scope into the module metadata.
        debug!("(resolving module) resolving module ID {}", id);
        visit::walk_mod(self, module, ());
    }

    fn resolve_local(&mut self, local: &Local) {
        // Resolve the type.
        self.resolve_type(local.ty);

        // Resolve the initializer, if necessary.
        match local.init {
            None => {
                // Nothing to do.
            }
            Some(initializer) => {
                self.resolve_expr(initializer);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern(local.pat, LocalIrrefutableMode, None);
    }

    // build a map from pattern identifiers to binding-info's.
    // this is done hygienically. This could arise for a macro
    // that expands into an or-pattern where one 'x' was from the
    // user and one 'x' came from the macro.
    fn binding_mode_map(&mut self, pat: &Pat) -> BindingMap {
        let mut result = HashMap::new();
        pat_bindings(&self.def_map, pat, |binding_mode, _id, sp, path| {
            let name = mtwt::resolve(path_to_ident(path));
            result.insert(name,
                          binding_info {span: sp,
                                        binding_mode: binding_mode});
        });
        return result;
    }

    // check that all of the arms in an or-pattern have exactly the
    // same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, arm: &Arm) {
        if arm.pats.len() == 0 {
            return
        }
        let map_0 = self.binding_mode_map(*arm.pats.get(0));
        for (i, p) in arm.pats.iter().enumerate() {
            let map_i = self.binding_mode_map(*p);

            for (&key, &binding_0) in map_0.iter() {
                match map_i.find(&key) {
                  None => {
                    self.resolve_error(
                        p.span,
                        format!("variable `{}` from pattern \\#1 is \
                                  not bound in pattern \\#{}",
                                token::get_name(key),
                                i + 1));
                  }
                  Some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.resolve_error(
                            binding_i.span,
                            format!("variable `{}` is bound with different \
                                      mode in pattern \\#{} than in pattern \\#1",
                                    token::get_name(key),
                                    i + 1));
                    }
                  }
                }
            }

            for (&key, &binding) in map_i.iter() {
                if !map_0.contains_key(&key) {
                    self.resolve_error(
                        binding.span,
                        format!("variable `{}` from pattern \\#{} is \
                                  not bound in pattern \\#1",
                                token::get_name(key),
                                i + 1));
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        self.value_ribs.borrow_mut().push(Rib::new(NormalRibKind));

        let mut bindings_list = HashMap::new();
        for pattern in arm.pats.iter() {
            self.resolve_pattern(*pattern,
                                 RefutableMode,
                                 Some(&mut bindings_list));
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        visit::walk_expr_opt(self, arm.guard, ());
        self.resolve_expr(arm.body);

        self.value_ribs.borrow_mut().pop();
    }

    fn resolve_block(&mut self, block: &Block) {
        debug!("(resolving block) entering block");
        self.value_ribs.borrow_mut().push(Rib::new(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module.clone();
        match orig_module.anonymous_children.borrow().find(&block.id) {
            None => { /* Nothing to do. */ }
            Some(anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module.clone();
            }
        }

        // Descend into the block.
        visit::walk_block(self, block, ());

        // Move back up.
        self.current_module = orig_module;

        self.value_ribs.borrow_mut().pop();
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            TyPath(ref path, ref bounds, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // looking for it.
                let mut result_def = None;

                // First, check to see whether the name is a primitive type.
                if path.segments.len() == 1 {
                    let id = path.segments.last().unwrap().identifier;

                    match self.primitive_type_table
                            .primitive_types
                            .find(&id.name) {

                        Some(&primitive_type) => {
                            result_def =
                                Some((DefPrimTy(primitive_type), LastMod(AllPublic)));

                            if path.segments
                                   .iter()
                                   .any(|s| !s.lifetimes.is_empty()) {
                                self.session.span_err(path.span,
                                                      "lifetime parameters \
                                                       are not allowed on \
                                                       this type")
                            } else if path.segments
                                          .iter()
                                          .any(|s| s.types.len() > 0) {
                                self.session.span_err(path.span,
                                                      "type parameters are \
                                                       not allowed on this \
                                                       type")
                            }
                        }
                        None => {
                            // Continue.
                        }
                    }
                }

                match result_def {
                    None => {
                        match self.resolve_path(ty.id, path, TypeNS, true) {
                            Some(def) => {
                                debug!("(resolving type) resolved `{}` to \
                                        type {:?}",
                                       token::get_ident(path.segments
                                                            .last().unwrap()
                                                            .identifier),
                                       def);
                                result_def = Some(def);
                            }
                            None => {
                                result_def = None;
                            }
                        }
                    }
                    Some(_) => {}   // Continue.
                }

                match result_def {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving type) writing resolution for `{}` \
                                (id {})",
                               self.path_idents_to_str(path),
                               path_id);
                        self.record_def(path_id, def);
                    }
                    None => {
                        let msg = format!("use of undeclared type name `{}`",
                                          self.path_idents_to_str(path));
                        self.resolve_error(ty.span, msg);
                    }
                }

                bounds.as_ref().map(|bound_vec| {
                    for bound in bound_vec.iter() {
                        self.resolve_type_parameter_bound(ty.id, bound);
                    }
                });
            }

            TyClosure(c, _) | TyProc(c) => {
                c.bounds.as_ref().map(|bounds| {
                    for bound in bounds.iter() {
                        self.resolve_type_parameter_bound(ty.id, bound);
                    }
                });
                visit::walk_ty(self, ty, ());
            }

            _ => {
                // Just resolve embedded types.
                visit::walk_ty(self, ty, ());
            }
        }
    }

    fn resolve_pattern(&mut self,
                       pattern: &Pat,
                       mode: PatternBindingMode,
                       // Maps idents to the node ID for the (outermost)
                       // pattern that binds them
                       mut bindings_list: Option<&mut HashMap<Name,NodeId>>) {
        let pat_id = pattern.id;
        walk_pat(pattern, |pattern| {
            match pattern.node {
                PatIdent(binding_mode, ref path, _)
                        if !path.global && path.segments.len() == 1 => {

                    // The meaning of pat_ident with no type parameters
                    // depends on whether an enum variant or unit-like struct
                    // with that name is in scope. The probing lookup has to
                    // be careful not to emit spurious errors. Only matching
                    // patterns (match) can match nullary variants or
                    // unit-like structs. For binding patterns (let), matching
                    // such a value is simply disallowed (since it's rarely
                    // what you want).

                    let ident = path.segments.get(0).identifier;
                    let renamed = mtwt::resolve(ident);

                    match self.resolve_bare_identifier_pattern(ident) {
                        FoundStructOrEnumVariant(def, lp)
                                if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    struct or enum variant",
                                   token::get_name(renamed));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "an enum variant");
                            self.record_def(pattern.id, (def, lp));
                        }
                        FoundStructOrEnumVariant(..) => {
                            self.resolve_error(pattern.span,
                                                  format!("declaration of `{}` \
                                                        shadows an enum \
                                                        variant or unit-like \
                                                        struct in scope",
                                                        token::get_name(renamed)));
                        }
                        FoundConst(def, lp) if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    constant",
                                   token::get_name(renamed));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "a constant");
                            self.record_def(pattern.id, (def, lp));
                        }
                        FoundConst(..) => {
                            self.resolve_error(pattern.span,
                                                  "only irrefutable patterns \
                                                   allowed here");
                        }
                        BareIdentifierPatternUnresolved => {
                            debug!("(resolving pattern) binding `{}`",
                                   token::get_name(renamed));

                            let def = match mode {
                                RefutableMode => {
                                    // For pattern arms, we must use
                                    // `def_binding` definitions.

                                    DefBinding(pattern.id, binding_mode)
                                }
                                LocalIrrefutableMode => {
                                    // But for locals, we use `def_local`.
                                    DefLocal(pattern.id, binding_mode)
                                }
                                ArgumentIrrefutableMode => {
                                    // And for function arguments, `def_arg`.
                                    DefArg(pattern.id, binding_mode)
                                }
                            };

                            // Record the definition so that later passes
                            // will be able to distinguish variants from
                            // locals in patterns.

                            self.record_def(pattern.id, (def, LastMod(AllPublic)));

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)

                            match bindings_list {
                                Some(ref mut bindings_list)
                                if !bindings_list.contains_key(&renamed) => {
                                    let this = &mut *self;
                                    let value_ribs = this.value_ribs.borrow();
                                    let length = value_ribs.len();
                                    let last_rib = value_ribs.get(
                                        length - 1);
                                    last_rib.bindings.borrow_mut()
                                            .insert(renamed, DlDef(def));
                                    bindings_list.insert(renamed, pat_id);
                                }
                                Some(ref mut b) => {
                                  if b.find(&renamed) == Some(&pat_id) {
                                      // Then this is a duplicate variable
                                      // in the same disjunct, which is an
                                      // error
                                     self.resolve_error(pattern.span,
                                       format!("identifier `{}` is bound more \
                                             than once in the same pattern",
                                            path_to_str(path)));
                                  }
                                  // Not bound in the same pattern: do nothing
                                }
                                None => {
                                    let this = &mut *self;
                                    {
                                        let value_ribs = this.value_ribs.borrow();
                                        let length = value_ribs.len();
                                        let last_rib = value_ribs.get(
                                                length - 1);
                                        last_rib.bindings.borrow_mut()
                                                .insert(renamed, DlDef(def));
                                    }
                                }
                            }
                        }
                    }

                    // Check the types in the path pattern.
                    for &ty in path.segments
                                  .iter()
                                  .flat_map(|seg| seg.types.iter()) {
                        self.resolve_type(ty);
                    }
                }

                PatIdent(binding_mode, ref path, _) => {
                    // This must be an enum variant, struct, or constant.
                    match self.resolve_path(pat_id, path, ValueNS, false) {
                        Some(def @ (DefVariant(..), _)) |
                        Some(def @ (DefStruct(..), _)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some(def @ (DefStatic(..), _)) => {
                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "a constant");
                            self.record_def(pattern.id, def);
                        }
                        Some(_) => {
                            self.resolve_error(
                                path.span,
                                format!("`{}` is not an enum variant or constant",
                                     token::get_ident(
                                         path.segments.last().unwrap().identifier)))
                        }
                        None => {
                            self.resolve_error(path.span,
                                                  "unresolved enum variant");
                        }
                    }

                    // Check the types in the path pattern.
                    for &ty in path.segments
                                  .iter()
                                  .flat_map(|s| s.types.iter()) {
                        self.resolve_type(ty);
                    }
                }

                PatEnum(ref path, _) => {
                    // This must be an enum variant, struct or const.
                    match self.resolve_path(pat_id, path, ValueNS, false) {
                        Some(def @ (DefFn(..), _))      |
                        Some(def @ (DefVariant(..), _)) |
                        Some(def @ (DefStruct(..), _))  |
                        Some(def @ (DefStatic(..), _)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some(_) => {
                            self.resolve_error(path.span,
                                format!("`{}` is not an enum variant, struct or const",
                                    token::get_ident(path.segments
                                                         .last().unwrap()
                                                         .identifier)));
                        }
                        None => {
                            self.resolve_error(path.span,
                                format!("unresolved enum variant, struct or const `{}`",
                                    token::get_ident(path.segments
                                                         .last().unwrap()
                                                         .identifier)));
                        }
                    }

                    // Check the types in the path pattern.
                    for &ty in path.segments
                                  .iter()
                                  .flat_map(|s| s.types.iter()) {
                        self.resolve_type(ty);
                    }
                }

                PatLit(expr) => {
                    self.resolve_expr(expr);
                }

                PatRange(first_expr, last_expr) => {
                    self.resolve_expr(first_expr);
                    self.resolve_expr(last_expr);
                }

                PatStruct(ref path, _, _) => {
                    match self.resolve_path(pat_id, path, TypeNS, false) {
                        Some((DefTy(class_id), lp))
                                if self.structs.contains_key(&class_id) => {
                            let class_def = DefStruct(class_id);
                            self.record_def(pattern.id, (class_def, lp));
                        }
                        Some(definition @ (DefStruct(class_id), _)) => {
                            assert!(self.structs.contains_key(&class_id));
                            self.record_def(pattern.id, definition);
                        }
                        Some(definition @ (DefVariant(_, variant_id, _), _))
                                if self.structs.contains_key(&variant_id) => {
                            self.record_def(pattern.id, definition);
                        }
                        result => {
                            debug!("(resolving pattern) didn't find struct \
                                    def: {:?}", result);
                            let msg = format!("`{}` does not name a structure",
                                              self.path_idents_to_str(path));
                            self.resolve_error(path.span, msg);
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

    fn resolve_bare_identifier_pattern(&mut self, name: Ident)
                                       -> BareIdentifierPatternResolution {
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module,
                                                 name,
                                                 ValueNS) {
            Success((target, _)) => {
                debug!("(resolve bare identifier pattern) succeeded in \
                         finding {} at {:?}",
                        token::get_ident(name),
                        target.bindings.value_def.borrow());
                match *target.bindings.value_def.borrow() {
                    None => {
                        fail!("resolved name in the value namespace to a \
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
                            def @ DefStatic(_, false) => {
                                return FoundConst(def, LastMod(AllPublic));
                            }
                            _ => {
                                return BareIdentifierPatternUnresolved;
                            }
                        }
                    }
                }
            }

            Indeterminate => {
                fail!("unexpected indeterminate result");
            }

            Failed => {
                debug!("(resolve bare identifier pattern) failed to find {}",
                        token::get_ident(name));
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
        // First, resolve the types.
        for &ty in path.segments.iter().flat_map(|s| s.types.iter()) {
            self.resolve_type(ty);
        }

        if path.global {
            return self.resolve_crate_relative_path(path, namespace);
        }

        let unqualified_def =
                self.resolve_identifier(path.segments
                                            .last().unwrap()
                                            .identifier,
                                        namespace,
                                        check_ribs,
                                        path.span);

        if path.segments.len() > 1 {
            let def = self.resolve_module_relative_path(path, namespace);
            match (def, unqualified_def) {
                (Some((d, _)), Some((ud, _))) if d == ud => {
                    self.session
                        .add_lint(UnnecessaryQualification,
                                  id,
                                  path.span,
                                  "unnecessary qualification".to_strbuf());
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

        return self.resolve_item_by_identifier_in_lexical_scope(identifier,
                                                                namespace);
    }

    // FIXME #4952: Merge me with resolve_name_in_module?
    fn resolve_definition_of_name_in_module(&mut self,
                                            containing_module: Rc<Module>,
                                            name: Name,
                                            namespace: Namespace)
                                                -> NameDefinition {
        // First, search children.
        self.populate_module_if_necessary(&containing_module);

        match containing_module.children.borrow().find(&name) {
            Some(child_name_bindings) => {
                match child_name_bindings.def_for_namespace(namespace) {
                    Some(def) => {
                        // Found it. Stop the search here.
                        let p = child_name_bindings.defined_in_public_namespace(
                                        namespace);
                        let lp = if p {LastMod(AllPublic)} else {
                            LastMod(DependsOn(def_id_of_def(def)))
                        };
                        return ChildNameDefinition(def, lp);
                    }
                    None => {}
                }
            }
            None => {}
        }

        // Next, search import resolutions.
        match containing_module.import_resolutions.borrow().find(&name) {
            Some(import_resolution) if import_resolution.is_public => {
                match (*import_resolution).target_for_namespace(namespace) {
                    Some(target) => {
                        match target.bindings.def_for_namespace(namespace) {
                            Some(def) => {
                                // Found it.
                                let id = import_resolution.id(namespace);
                                self.used_imports.insert((id, namespace));
                                return ImportNameDefinition(def, LastMod(AllPublic));
                            }
                            None => {
                                // This can happen with external impls, due to
                                // the imperfect way we read the metadata.
                            }
                        }
                    }
                    None => {}
                }
            }
            Some(..) | None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            match containing_module.external_module_children.borrow()
                                   .find_copy(&name) {
                None => {}
                Some(module) => {
                    match module.def_id.get() {
                        None => {} // Continue.
                        Some(def_id) => {
                            let lp = if module.is_public {LastMod(AllPublic)} else {
                                LastMod(DependsOn(def_id))
                            };
                            return ChildNameDefinition(DefMod(def_id), lp);
                        }
                    }
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
        let module_path_idents = path.segments.init().iter()
                                                     .map(|ps| ps.identifier)
                                                     .collect::<Vec<_>>();

        let containing_module;
        let last_private;
        let module = self.current_module.clone();
        match self.resolve_module_path(module,
                                       module_path_idents.as_slice(),
                                       UseLexicalScope,
                                       path.span,
                                       PathSearch) {
            Failed => {
                let msg = format!("use of undeclared module `{}`",
                                  self.idents_to_str(module_path_idents.as_slice()));
                self.resolve_error(path.span, msg);
                return None;
            }

            Indeterminate => {
                fail!("indeterminate unexpected");
            }

            Success((resulting_module, resulting_last_private)) => {
                containing_module = resulting_module;
                last_private = resulting_last_private;
            }
        }

        let ident = path.segments.last().unwrap().identifier;
        let def = match self.resolve_definition_of_name_in_module(containing_module.clone(),
                                                        ident.name,
                                                        namespace) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                return None;
            }
            ChildNameDefinition(def, lp) | ImportNameDefinition(def, lp) => {
                (def, last_private.or(lp))
            }
        };
        match containing_module.kind.get() {
            TraitModuleKind | ImplModuleKind => {
                match containing_module.def_id.get() {
                    Some(def_id) => {
                        match self.method_map.borrow().find(&(ident.name, def_id)) {
                            Some(x) if *x == SelfStatic => (),
                            None => (),
                            _ => {
                                debug!("containing module was a trait or impl \
                                and name was a method -> not resolved");
                                return None;
                            }
                        }
                    },
                    _ => (),
                }
            },
            _ => (),
        }
        return Some(def);
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path(&mut self,
                                   path: &Path,
                                   namespace: Namespace)
                                       -> Option<(Def, LastPrivate)> {
        let module_path_idents = path.segments.init().iter()
                                                     .map(|ps| ps.identifier)
                                                     .collect::<Vec<_>>();

        let root_module = self.graph_root.get_module();

        let containing_module;
        let last_private;
        match self.resolve_module_path_from_root(root_module,
                                                 module_path_idents.as_slice(),
                                                 0,
                                                 path.span,
                                                 PathSearch,
                                                 LastMod(AllPublic)) {
            Failed => {
                let msg = format!("use of undeclared module `::{}`",
                                  self.idents_to_str(module_path_idents.as_slice()));
                self.resolve_error(path.span, msg);
                return None;
            }

            Indeterminate => {
                fail!("indeterminate unexpected");
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
                self.search_ribs(self.value_ribs.borrow().as_slice(),
                                 renamed, span)
            }
            TypeNS => {
                let name = ident.name;
                self.search_ribs(self.type_ribs.borrow().as_slice(), name, span)
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

    fn resolve_item_by_identifier_in_lexical_scope(&mut self,
                                                   ident: Ident,
                                                   namespace: Namespace)
                                                -> Option<(Def, LastPrivate)> {
        // Check the items.
        let module = self.current_module.clone();
        match self.resolve_item_in_lexical_scope(module,
                                                 ident,
                                                 namespace) {
            Success((target, _)) => {
                match (*target.bindings).def_for_namespace(namespace) {
                    None => {
                        // This can happen if we were looking for a type and
                        // found a module instead. Modules don't have defs.
                        debug!("(resolving item path by identifier in lexical \
                                 scope) failed to resolve {} after success...",
                                 token::get_ident(ident));
                        return None;
                    }
                    Some(def) => {
                        debug!("(resolving item path in lexical scope) \
                                resolved `{}` to item",
                               token::get_ident(ident));
                        // This lookup is "all public" because it only searched
                        // for one identifier in the current module (couldn't
                        // have passed through reexports or anything like that.
                        return Some((def, LastMod(AllPublic)));
                    }
                }
            }
            Indeterminate => {
                fail!("unexpected indeterminate result");
            }
            Failed => {
                debug!("(resolving item path by identifier in lexical scope) \
                         failed to resolve {}", token::get_ident(ident));
                return None;
            }
        }
    }

    fn with_no_errors<T>(&mut self, f: |&mut Resolver| -> T) -> T {
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
        fn get_module(this: &mut Resolver, span: Span, ident_path: &[ast::Ident])
                            -> Option<Rc<Module>> {
            let root = this.current_module.clone();
            let last_name = ident_path.last().unwrap().name;

            if ident_path.len() == 1 {
                match this.primitive_type_table.primitive_types.find(&last_name) {
                    Some(_) => None,
                    None => {
                        match this.current_module.children.borrow().find(&last_name) {
                            Some(child) => child.get_module_if_available(),
                            None => None
                        }
                    }
                }
            } else {
                match this.resolve_module_path(root,
                                                ident_path.as_slice(),
                                                UseLexicalScope,
                                                span,
                                                PathSearch) {
                    Success((module, _)) => Some(module),
                    _ => None
                }
            }
        }

        let (path, node_id) = match self.current_self_type {
            Some(ref ty) => match ty.node {
                TyPath(ref path, _, node_id) => (path.clone(), node_id),
                _ => unreachable!(),
            },
            None => return NoSuggestion,
        };

        // Look for a field with the same name in the current self_type.
        match self.def_map.borrow().find(&node_id) {
             Some(&DefTy(did))
            | Some(&DefStruct(did))
            | Some(&DefVariant(_, did, _)) => match self.structs.find(&did) {
                None => {}
                Some(fields) => {
                    if fields.iter().any(|&field_name| name == field_name) {
                        return Field;
                    }
                }
            },
            _ => {} // Self type didn't resolve properly
        }

        let ident_path = path.segments.iter().map(|seg| seg.identifier).collect::<Vec<_>>();

        // Look for a method in the current self type's impl module.
        match get_module(self, path.span, ident_path.as_slice()) {
            Some(module) => match module.children.borrow().find(&name) {
                Some(binding) => {
                    let p_str = self.path_idents_to_str(&path);
                    match binding.def_for_namespace(ValueNS) {
                        Some(DefStaticMethod(_, provenance, _)) => {
                            match provenance {
                                FromImpl(_) => return StaticMethod(p_str),
                                FromTrait(_) => unreachable!()
                            }
                        }
                        Some(DefMethod(_, None)) => return Method,
                        Some(DefMethod(_, _)) => return TraitMethod,
                        _ => ()
                    }
                }
                None => {}
            },
            None => {}
        }

        // Look for a method in the current trait.
        let method_map = self.method_map.borrow();
        match self.current_trait_ref {
            Some((did, ref trait_ref)) => {
                let path_str = self.path_idents_to_str(&trait_ref.path);

                match method_map.find(&(name, did)) {
                    Some(&SelfStatic) => return StaticTraitMethod(path_str),
                    Some(_) => return TraitMethod,
                    None => {}
                }
            }
            None => {}
        }

        NoSuggestion
    }

    fn find_best_match_for_name(&mut self, name: &str, max_distance: uint)
                                -> Option<StrBuf> {
        let this = &mut *self;

        let mut maybes: Vec<token::InternedString> = Vec::new();
        let mut values: Vec<uint> = Vec::new();

        let mut j = this.value_ribs.borrow().len();
        while j != 0 {
            j -= 1;
            let value_ribs = this.value_ribs.borrow();
            let bindings = value_ribs.get(j).bindings.borrow();
            for (&k, _) in bindings.iter() {
                maybes.push(token::get_name(k));
                values.push(uint::MAX);
            }
        }

        let mut smallest = 0;
        for (i, other) in maybes.iter().enumerate() {
            *values.get_mut(i) = name.lev_distance(other.get());

            if *values.get(i) <= *values.get(smallest) {
                smallest = i;
            }
        }

        if values.len() > 0 &&
            *values.get(smallest) != uint::MAX &&
            *values.get(smallest) < name.len() + 2 &&
            *values.get(smallest) <= max_distance &&
            name != maybes.get(smallest).get() {

            Some(maybes.get(smallest).get().to_strbuf())

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

            ExprPath(ref path) => {
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.

                match self.resolve_path(expr.id, path, ValueNS, true) {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving expr) resolved `{}`",
                               self.path_idents_to_str(path));

                        // First-class methods are not supported yet; error
                        // out here.
                        match def {
                            (DefMethod(..), _) => {
                                self.resolve_error(expr.span,
                                                      "first-class methods \
                                                       are not supported");
                                self.session.span_note(expr.span,
                                                       "call the method \
                                                        using the `.` \
                                                        syntax");
                            }
                            _ => {}
                        }

                        self.record_def(expr.id, def);
                    }
                    None => {
                        let wrong_name = self.path_idents_to_str(path);
                        // Be helpful if the name refers to a struct
                        // (The pattern matching def_tys where the id is in self.structs
                        // matches on regular structs while excluding tuple- and enum-like
                        // structs, which wouldn't result in this error.)
                        match self.with_no_errors(|this|
                            this.resolve_path(expr.id, path, TypeNS, false)) {
                            Some((DefTy(struct_id), _))
                              if self.structs.contains_key(&struct_id) => {
                                self.resolve_error(expr.span,
                                        format!("`{}` is a structure name, but \
                                                 this expression \
                                                 uses it like a function name",
                                                wrong_name));

                                self.session.span_note(expr.span,
                                    format!("Did you mean to write: \
                                            `{} \\{ /* fields */ \\}`?",
                                            wrong_name));

                            }
                            _ => {
                                let mut method_scope = false;
                                self.value_ribs.borrow().iter().rev().advance(|rib| {
                                    let res = match *rib {
                                        Rib { bindings: _, kind: MethodRibKind(_, _) } => true,
                                        Rib { bindings: _, kind: OpaqueFunctionRibKind } => false,
                                        _ => return true, // Keep advancing
                                    };

                                    method_scope = res;
                                    false // Stop advancing
                                });

                                if method_scope && token::get_name(self.self_ident.name).get()
                                                                        == wrong_name.as_slice() {
                                        self.resolve_error(expr.span,
                                                            format!("`self` is not available in a \
                                                                    static method. Maybe a `self` \
                                                                    argument is missing?"));
                                } else {
                                    let name = path_to_ident(path).name;
                                    let mut msg = match self.find_fallback_in_self_type(name) {
                                        NoSuggestion => {
                                            // limit search to 5 to reduce the number
                                            // of stupid suggestions
                                            self.find_best_match_for_name(wrong_name.as_slice(), 5)
                                                                .map_or("".into_owned(),
                                                                        |x| format!("`{}`", x))
                                        }
                                        Field =>
                                            format!("`self.{}`", wrong_name),
                                        Method
                                        | TraitMethod =>
                                            format!("to call `self.{}`", wrong_name),
                                        StaticTraitMethod(path_str)
                                        | StaticMethod(path_str) =>
                                            format!("to call `{}::{}`", path_str, wrong_name)
                                    };

                                    if msg.len() > 0 {
                                        msg = format!(" Did you mean {}?", msg)
                                    }

                                    self.resolve_error(expr.span, format!("unresolved name `{}`.{}",
                                                                            wrong_name, msg));
                                }
                            }
                        }
                    }
                }

                visit::walk_expr(self, expr, ());
            }

            ExprFnBlock(fn_decl, block) |
            ExprProc(fn_decl, block) => {
                self.resolve_function(FunctionRibKind(expr.id, block.id),
                                      Some(fn_decl), NoTypeParameters,
                                      block);
            }

            ExprStruct(ref path, _, _) => {
                // Resolve the path to the structure it goes to.
                match self.resolve_path(expr.id, path, TypeNS, false) {
                    Some((DefTy(class_id), lp)) | Some((DefStruct(class_id), lp))
                            if self.structs.contains_key(&class_id) => {
                        let class_def = DefStruct(class_id);
                        self.record_def(expr.id, (class_def, lp));
                    }
                    Some(definition @ (DefVariant(_, class_id, _), _))
                            if self.structs.contains_key(&class_id) => {
                        self.record_def(expr.id, definition);
                    }
                    result => {
                        debug!("(resolving expression) didn't find struct \
                                def: {:?}", result);
                        let msg = format!("`{}` does not name a structure",
                                          self.path_idents_to_str(path));
                        self.resolve_error(path.span, msg);
                    }
                }

                visit::walk_expr(self, expr, ());
            }

            ExprLoop(_, Some(label)) => {
                self.with_label_rib(|this| {
                    let def_like = DlDef(DefLabel(expr.id));

                    {
                        let label_ribs = this.label_ribs.borrow();
                        let length = label_ribs.len();
                        let rib = label_ribs.get(length - 1);
                        let renamed = mtwt::resolve(label);
                        rib.bindings.borrow_mut().insert(renamed, def_like);
                    }

                    visit::walk_expr(this, expr, ());
                })
            }

            ExprForLoop(..) => fail!("non-desugared expr_for_loop"),

            ExprBreak(Some(label)) | ExprAgain(Some(label)) => {
                let renamed = mtwt::resolve(label);
                match self.search_ribs(self.label_ribs.borrow().as_slice(),
                                       renamed, expr.span) {
                    None =>
                        self.resolve_error(expr.span,
                                              format!("use of undeclared label `{}`",
                                                   token::get_ident(label))),
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
                visit::walk_expr(self, expr, ());
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self, expr: &Expr) {
        match expr.node {
            ExprField(_, ident, _) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.search_for_traits_containing_method(ident.name);
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
                    let method_map = self.method_map.borrow();

                    if method_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
                None => {} // Nothing to do.
            }

            // Look for trait children.
            self.populate_module_if_necessary(&search_module);

            {
                let method_map = self.method_map.borrow();
                for (_, child_names) in search_module.children.borrow().iter() {
                    let def = match child_names.def_for_namespace(TypeNS) {
                        Some(def) => def,
                        None => continue
                    };
                    let trait_def_id = match def {
                        DefTrait(trait_def_id) => trait_def_id,
                        _ => continue,
                    };
                    if method_map.contains_key(&(name, trait_def_id)) {
                        add_trait_info(&mut found_traits, trait_def_id, name);
                    }
                }
            }

            // Look for imports.
            for (_, import) in search_module.import_resolutions.borrow().iter() {
                let target = match import.target_for_namespace(TypeNS) {
                    None => continue,
                    Some(target) => target,
                };
                let did = match target.bindings.def_for_namespace(TypeNS) {
                    Some(DefTrait(trait_def_id)) => trait_def_id,
                    Some(..) | None => continue,
                };
                if self.method_map.borrow().contains_key(&(name, did)) {
                    add_trait_info(&mut found_traits, did, name);
                    self.used_imports.insert((import.type_id, TypeNS));
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
        debug!("(recording def) recording {:?} for {:?}, last private {:?}",
                def, node_id, lp);
        assert!(match lp {LastImport{..} => false, _ => true},
                "Import should only be used for `use` directives");
        self.last_private.insert(node_id, lp);
        self.def_map.borrow_mut().insert_or_update_with(node_id, def, |_, old_value| {
            // Resolve appears to "resolve" the same ID multiple
            // times, so here is a sanity check it at least comes to
            // the same conclusion! - nmatsakis
            if def != *old_value {
                self.session.bug(format!("node_id {:?} resolved first to {:?} \
                                      and then {:?}", node_id, *old_value, def));
            }
        });
    }

    fn enforce_default_binding_mode(&mut self,
                                        pat: &Pat,
                                        pat_binding_mode: BindingMode,
                                        descr: &str) {
        match pat_binding_mode {
            BindByValue(_) => {}
            BindByRef(..) => {
                self.resolve_error(
                    pat.span,
                    format!("cannot use `ref` binding mode with {}",
                         descr));
            }
        }
    }

    //
    // Unused import checking
    //
    // Although this is mostly a lint pass, it lives in here because it depends on
    // resolve data structures and because it finalises the privacy information for
    // `use` directives.
    //

    fn check_for_unused_imports(&mut self, krate: &ast::Crate) {
        let mut visitor = UnusedImportCheckVisitor{ resolver: self };
        visit::walk_crate(&mut visitor, krate, ());
    }

    fn check_for_item_unused_imports(&mut self, vi: &ViewItem) {
        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if vi.vis == Public { return }
        if vi.span == DUMMY_SP { return }

        match vi.node {
            ViewItemExternCrate(..) => {} // ignore
            ViewItemUse(ref p) => {
                match p.node {
                    ViewPathSimple(_, _, id) => self.finalize_import(id, p.span),
                    ViewPathList(_, ref list, _) => {
                        for i in list.iter() {
                            self.finalize_import(i.node.id, i.span);
                        }
                    },
                    ViewPathGlob(_, id) => {
                        if !self.used_imports.contains(&(id, TypeNS)) &&
                           !self.used_imports.contains(&(id, ValueNS)) {
                            self.session
                                .add_lint(UnusedImports,
                                          id,
                                          p.span,
                                          "unused import".to_strbuf());
                        }
                    },
                }
            }
        }
    }

    // We have information about whether `use` (import) directives are actually used now.
    // If an import is not used at all, we signal a lint error. If an import is only used
    // for a single namespace, we remove the other namespace from the recorded privacy
    // information. That means in privacy.rs, we will only check imports and namespaces
    // which are used. In particular, this means that if an import could name either a
    // public or private item, we will check the correct thing, dependent on how the import
    // is used.
    fn finalize_import(&mut self, id: NodeId, span: Span) {
        debug!("finalizing import uses for {}",
               self.session.codemap().span_to_snippet(span));

        if !self.used_imports.contains(&(id, TypeNS)) &&
           !self.used_imports.contains(&(id, ValueNS)) {
            self.session.add_lint(UnusedImports,
                                  id,
                                  span,
                                  "unused import".to_strbuf());
        }

        let (v_priv, t_priv) = match self.last_private.find(&id) {
            Some(&LastImport {
                value_priv: v,
                value_used: _,
                type_priv: t,
                type_used: _
            }) => (v, t),
            Some(_) => {
                fail!("we should only have LastImport for `use` directives")
            }
            _ => return,
        };

        let mut v_used = if self.used_imports.contains(&(id, ValueNS)) {
            Used
        } else {
            Unused
        };
        let t_used = if self.used_imports.contains(&(id, TypeNS)) {
            Used
        } else {
            Unused
        };

        match (v_priv, t_priv) {
            // Since some items may be both in the value _and_ type namespaces (e.g., structs)
            // we might have two LastPrivates pointing at the same thing. There is no point
            // checking both, so lets not check the value one.
            (Some(DependsOn(def_v)), Some(DependsOn(def_t))) if def_v == def_t => v_used = Unused,
            _ => {},
        }

        self.last_private.insert(id, LastImport{value_priv: v_priv,
                                                value_used: v_used,
                                                type_priv: t_priv,
                                                type_used: t_used});
    }

    //
    // Diagnostics
    //
    // Diagnostics are not particularly efficient, because they're rarely
    // hit.
    //

    /// A somewhat inefficient routine to obtain the name of a module.
    fn module_to_str(&mut self, module: &Module) -> StrBuf {
        let mut idents = Vec::new();

        fn collect_mod(idents: &mut Vec<ast::Ident>, module: &Module) {
            match module.parent_link {
                NoParentLink => {}
                ModuleParentLink(ref module, name) => {
                    idents.push(name);
                    collect_mod(idents, &*module.upgrade().unwrap());
                }
                BlockParentLink(ref module, _) => {
                    idents.push(special_idents::opaque);
                    collect_mod(idents, &*module.upgrade().unwrap());
                }
            }
        }
        collect_mod(&mut idents, module);

        if idents.len() == 0 {
            return "???".to_strbuf();
        }
        self.idents_to_str(idents.move_iter().rev()
                                 .collect::<Vec<ast::Ident>>()
                                 .as_slice())
    }

    #[allow(dead_code)]   // useful for debugging
    fn dump_module(&mut self, module_: Rc<Module>) {
        debug!("Dump of module `{}`:", self.module_to_str(&*module_));

        debug!("Children:");
        self.populate_module_if_necessary(&module_);
        for (&name, _) in module_.children.borrow().iter() {
            debug!("* {}", token::get_name(name));
        }

        debug!("Import resolutions:");
        let import_resolutions = module_.import_resolutions.borrow();
        for (&name, import_resolution) in import_resolutions.iter() {
            let value_repr;
            match import_resolution.target_for_namespace(ValueNS) {
                None => { value_repr = "".to_owned(); }
                Some(_) => {
                    value_repr = " value:?".to_owned();
                    // FIXME #4954
                }
            }

            let type_repr;
            match import_resolution.target_for_namespace(TypeNS) {
                None => { type_repr = "".to_owned(); }
                Some(_) => {
                    type_repr = " type:?".to_owned();
                    // FIXME #4954
                }
            }

            debug!("* {}:{}{}", token::get_name(name), value_repr, type_repr);
        }
    }
}

pub struct CrateMap {
    pub def_map: DefMap,
    pub exp_map2: ExportMap2,
    pub trait_map: TraitMap,
    pub external_exports: ExternalExports,
    pub last_private_map: LastPrivateMap,
}

/// Entry point to crate resolution.
pub fn resolve_crate(session: &Session,
                     lang_items: &LanguageItems,
                     krate: &Crate)
                  -> CrateMap {
    let mut resolver = Resolver(session, lang_items, krate.span);
    resolver.resolve(krate);
    let Resolver { def_map, export_map2, trait_map, last_private,
                   external_exports, .. } = resolver;
    CrateMap {
        def_map: def_map,
        exp_map2: export_map2,
        trait_map: trait_map,
        external_exports: external_exports,
        last_private_map: last_private,
    }
}
