// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use driver::session::Session;
use metadata::csearch;
use metadata::decoder::{DefLike, DlDef, DlField, DlImpl};
use middle::lang_items::LanguageItems;
use middle::lint::{unnecessary_qualification, unused_imports};
use middle::pat_util::pat_bindings;

use syntax::ast::*;
use syntax::ast;
use syntax::ast_util::{def_id_of_def, local_def, mtwt_resolve};
use syntax::ast_util::{path_to_ident, walk_pat, trait_method_to_ty_method};
use syntax::parse::token;
use syntax::parse::token::{ident_interner, interner_get};
use syntax::parse::token::special_idents;
use syntax::print::pprust::path_to_str;
use syntax::codemap::{Span, DUMMY_SP, Pos};
use syntax::opt_vec::OptVec;
use syntax::visit;
use syntax::visit::Visitor;

use std::cell::{Cell, RefCell};
use std::uint;
use std::hashmap::{HashMap, HashSet};
use std::util;

// Definition mapping
pub type DefMap = @RefCell<HashMap<NodeId,Def>>;

struct binding_info {
    span: Span,
    binding_mode: BindingMode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<Name,binding_info>;

// Trait method resolution
pub type TraitMap = HashMap<NodeId,@RefCell<~[DefId]>>;

// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap2 = @RefCell<HashMap<NodeId, ~[Export2]>>;

pub struct Export2 {
    name: @str,        // The name of the target.
    def_id: DefId,     // The definition of the target.
    reexport: bool,     // Whether this is a reexport.
}

// This set contains all exported definitions from external crates. The set does
// not contain any entries from local crates.
pub type ExternalExports = HashSet<DefId>;

// XXX: dox
pub type LastPrivateMap = HashMap<NodeId, LastPrivate>;

pub enum LastPrivate {
    AllPublic,
    DependsOn(DefId),
}

impl LastPrivate {
    fn or(self, other: LastPrivate) -> LastPrivate {
        match (self, other) {
            (me, AllPublic) => me,
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

#[deriving(Eq)]
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
    BoundResult(@Module, @NameBindings)
}

impl NamespaceResult {
    fn is_unknown(&self) -> bool {
        match *self {
            UnknownResult => true,
            _ => false
        }
    }
}

enum NameDefinition {
    NoNameDefinition,           //< The name was unbound.
    ChildNameDefinition(Def, LastPrivate), //< The name identifies an immediate child.
    ImportNameDefinition(Def, LastPrivate) //< The name identifies an import.
}

enum SelfBinding {
    NoSelfBinding,
    HasSelfBinding(NodeId, explicit_self)
}

impl Visitor<()> for Resolver {
    fn visit_item(&mut self, item:@item, _:()) {
        self.resolve_item(item);
    }
    fn visit_arm(&mut self, arm:&Arm, _:()) {
        self.resolve_arm(arm);
    }
    fn visit_block(&mut self, block:P<Block>, _:()) {
        self.resolve_block(block);
    }
    fn visit_expr(&mut self, expr:@Expr, _:()) {
        self.resolve_expr(expr);
    }
    fn visit_local(&mut self, local:@Local, _:()) {
        self.resolve_local(local);
    }
    fn visit_ty(&mut self, ty:&Ty, _:()) {
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
    ModuleReducedGraphParent(@Module)
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

enum SearchThroughModulesFlag {
    DontSearchThroughModules,
    SearchThroughModules
}

enum ModulePrefixResult {
    NoPrefixFound,
    PrefixFound(@Module, uint)
}

#[deriving(Eq)]
enum AllowCapturingSelfFlag {
    AllowCapturingSelf,         //< The "self" definition can be captured.
    DontAllowCapturingSelf,     //< The "self" definition cannot be captured.
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
    self_binding: RefCell<Option<DefLike>>,
    kind: RibKind,
}

impl Rib {
    fn new(kind: RibKind) -> Rib {
        Rib {
            bindings: RefCell::new(HashMap::new()),
            self_binding: RefCell::new(None),
            kind: kind
        }
    }
}

/// One import directive.
struct ImportDirective {
    module_path: ~[Ident],
    subclass: @ImportDirectiveSubclass,
    span: Span,
    id: NodeId,
    is_public: bool, // see note in ImportResolution about how to use this
}

impl ImportDirective {
    fn new(module_path: ~[Ident],
           subclass: @ImportDirectiveSubclass,
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
    target_module: @Module,
    bindings: @NameBindings,
}

impl Target {
    fn new(target_module: @Module, bindings: @NameBindings) -> Target {
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
    is_public: Cell<bool>,

    // The number of outstanding references to this name. When this reaches
    // zero, outside modules can count on the targets being correct. Before
    // then, all bets are off; future imports could override this name.
    outstanding_references: Cell<uint>,

    /// The value that this `use` directive names, if there is one.
    value_target: RefCell<Option<Target>>,
    /// The source node of the `use` directive leading to the value target
    /// being non-none
    value_id: Cell<NodeId>,

    /// The type that this `use` directive names, if there is one.
    type_target: RefCell<Option<Target>>,
    /// The source node of the `use` directive leading to the type target
    /// being non-none
    type_id: Cell<NodeId>,
}

impl ImportResolution {
    fn new(id: NodeId, is_public: bool) -> ImportResolution {
        ImportResolution {
            type_id: Cell::new(id),
            value_id: Cell::new(id),
            outstanding_references: Cell::new(0),
            value_target: RefCell::new(None),
            type_target: RefCell::new(None),
            is_public: Cell::new(is_public),
        }
    }

    fn target_for_namespace(&self, namespace: Namespace)
                                -> Option<Target> {
        match namespace {
            TypeNS      => return self.type_target.get(),
            ValueNS     => return self.value_target.get(),
        }
    }

    fn id(&self, namespace: Namespace) -> NodeId {
        match namespace {
            TypeNS  => self.type_id.get(),
            ValueNS => self.value_id.get(),
        }
    }
}

/// The link from a module up to its nearest parent node.
enum ParentLink {
    NoParentLink,
    ModuleParentLink(@Module, Ident),
    BlockParentLink(@Module, NodeId)
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

    children: RefCell<HashMap<Name, @NameBindings>>,
    imports: RefCell<~[@ImportDirective]>,

    // The external module children of this node that were declared with
    // `extern mod`.
    external_module_children: RefCell<HashMap<Name, @Module>>,

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
    anonymous_children: RefCell<HashMap<NodeId,@Module>>,

    // The status of resolving each import in this module.
    import_resolutions: RefCell<HashMap<Name, @ImportResolution>>,

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
            imports: RefCell::new(~[]),
            external_module_children: RefCell::new(HashMap::new()),
            anonymous_children: RefCell::new(HashMap::new()),
            import_resolutions: RefCell::new(HashMap::new()),
            glob_count: Cell::new(0),
            resolved_import_count: Cell::new(0),
            populated: Cell::new(!external),
        }
    }

    fn all_imports_resolved(&self) -> bool {
        let mut imports = self.imports.borrow_mut();
        return imports.get().len() == self.resolved_import_count.get();
    }
}

// Records a possibly-private type definition.
#[deriving(Clone)]
struct TypeNsDef {
    is_public: bool, // see note in ImportResolution about how to use this
    module_def: Option<@Module>,
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
        let module_ = @Module::new(parent_link, def_id, kind, external,
                                       is_public);
        match self.type_def.get() {
            None => {
                self.type_def.set(Some(TypeNsDef {
                    is_public: is_public,
                    module_def: Some(module_),
                    type_def: None,
                    type_span: Some(sp)
                }));
            }
            Some(type_def) => {
                self.type_def.set(Some(TypeNsDef {
                    is_public: is_public,
                    module_def: Some(module_),
                    type_span: Some(sp),
                    type_def: type_def.type_def
                }));
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
        match self.type_def.get() {
            None => {
                let module = @Module::new(parent_link, def_id, kind,
                                              external, is_public);
                self.type_def.set(Some(TypeNsDef {
                    is_public: is_public,
                    module_def: Some(module),
                    type_def: None,
                    type_span: None,
                }))
            }
            Some(type_def) => {
                match type_def.module_def {
                    None => {
                        let module = @Module::new(parent_link,
                                                      def_id,
                                                      kind,
                                                      external,
                                                      is_public);
                        self.type_def.set(Some(TypeNsDef {
                            is_public: is_public,
                            module_def: Some(module),
                            type_def: type_def.type_def,
                            type_span: None,
                        }))
                    }
                    Some(module_def) => module_def.kind.set(kind),
                }
            }
        }
    }

    /// Records a type definition.
    fn define_type(&self, def: Def, sp: Span, is_public: bool) {
        // Merges the type with the existing type def or creates a new one.
        match self.type_def.get() {
            None => {
                self.type_def.set(Some(TypeNsDef {
                    module_def: None,
                    type_def: Some(def),
                    type_span: Some(sp),
                    is_public: is_public,
                }));
            }
            Some(type_def) => {
                self.type_def.set(Some(TypeNsDef {
                    type_def: Some(def),
                    type_span: Some(sp),
                    module_def: type_def.module_def,
                    is_public: is_public,
                }));
            }
        }
    }

    /// Records a value definition.
    fn define_value(&self, def: Def, sp: Span, is_public: bool) {
        self.value_def.set(Some(ValueNsDef {
            def: def,
            value_span: Some(sp),
            is_public: is_public,
        }));
    }

    /// Returns the module node if applicable.
    fn get_module_if_available(&self) -> Option<@Module> {
        let type_def = self.type_def.borrow();
        match *type_def.get() {
            Some(ref type_def) => (*type_def).module_def,
            None => None
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module(&self) -> @Module {
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
            TypeNS   => return self.type_def.get().is_some(),
            ValueNS  => return self.value_def.get().is_some()
        }
    }

    fn defined_in_public_namespace(&self, namespace: Namespace) -> bool {
        match namespace {
            TypeNS => match self.type_def.get() {
                Some(def) => def.is_public, None => false
            },
            ValueNS => match self.value_def.get() {
                Some(def) => def.is_public, None => false
            }
        }
    }

    fn def_for_namespace(&self, namespace: Namespace) -> Option<Def> {
        match namespace {
            TypeNS => {
                match self.type_def.get() {
                    None => None,
                    Some(type_def) => {
                        match type_def.type_def {
                            Some(type_def) => Some(type_def),
                            None => {
                                match type_def.module_def {
                                    Some(module) => {
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
                match self.value_def.get() {
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
                    match self.type_def.get() {
                        None => None,
                        Some(type_def) => type_def.type_span
                    }
                }
                ValueNS => {
                    match self.value_def.get() {
                        None => None,
                        Some(value_def) => value_def.value_span
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
    primitive_types: HashMap<Name,prim_ty>,
}

impl PrimitiveTypeTable {
    fn intern(&mut self,
                  string: &str,
                  primitive_type: prim_ty) {
        self.primitive_types.insert(token::intern(string), primitive_type);
    }
}

fn PrimitiveTypeTable() -> PrimitiveTypeTable {
    let mut table = PrimitiveTypeTable {
        primitive_types: HashMap::new()
    };

    table.intern("bool",    ty_bool);
    table.intern("char",    ty_char);
    table.intern("f32",     ty_float(ty_f32));
    table.intern("f64",     ty_float(ty_f64));
    table.intern("int",     ty_int(ty_i));
    table.intern("i8",      ty_int(ty_i8));
    table.intern("i16",     ty_int(ty_i16));
    table.intern("i32",     ty_int(ty_i32));
    table.intern("i64",     ty_int(ty_i64));
    table.intern("str",     ty_str);
    table.intern("uint",    ty_uint(ty_u));
    table.intern("u8",      ty_uint(ty_u8));
    table.intern("u16",     ty_uint(ty_u16));
    table.intern("u32",     ty_uint(ty_u32));
    table.intern("u64",     ty_uint(ty_u64));

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

fn Resolver(session: Session,
            lang_items: LanguageItems,
            crate_span: Span) -> Resolver {
    let graph_root = @NameBindings();

    graph_root.define_module(NoParentLink,
                             Some(DefId { crate: 0, node: 0 }),
                             NormalModuleKind,
                             false,
                             true,
                             crate_span);

    let current_module = graph_root.get_module();

    let this = Resolver {
        session: @session,
        lang_items: lang_items,

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        graph_root: graph_root,

        method_map: @RefCell::new(HashMap::new()),
        structs: HashSet::new(),

        unresolved_imports: 0,

        current_module: current_module,
        value_ribs: @RefCell::new(~[]),
        type_ribs: @RefCell::new(~[]),
        label_ribs: @RefCell::new(~[]),

        current_trait_refs: None,

        self_ident: special_idents::self_,
        type_self_ident: special_idents::type_self,

        primitive_type_table: @PrimitiveTypeTable(),

        namespaces: ~[ TypeNS, ValueNS ],

        def_map: @RefCell::new(HashMap::new()),
        export_map2: @RefCell::new(HashMap::new()),
        trait_map: HashMap::new(),
        used_imports: HashSet::new(),
        external_exports: HashSet::new(),
        last_private: HashMap::new(),

        emit_errors: true,
        intr: session.intr()
    };

    this
}

/// The main resolver class.
struct Resolver {
    session: @Session,
    lang_items: LanguageItems,

    intr: @ident_interner,

    graph_root: @NameBindings,

    method_map: @RefCell<HashMap<Name, HashSet<DefId>>>,
    structs: HashSet<DefId>,

    // The number of imports that are currently unresolved.
    unresolved_imports: uint,

    // The module that represents the current item scope.
    current_module: @Module,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: @RefCell<~[@Rib]>,

    // The current set of local scopes, for types.
    type_ribs: @RefCell<~[@Rib]>,

    // The current set of local scopes, for labels.
    label_ribs: @RefCell<~[@Rib]>,

    // The trait that the current context can refer to.
    current_trait_refs: Option<~[DefId]>,

    // The ident for the keyword "self".
    self_ident: Ident,
    // The ident for the non-keyword "Self".
    type_self_ident: Ident,

    // The idents for the primitive types.
    primitive_type_table: @PrimitiveTypeTable,

    // The four namespaces.
    namespaces: ~[Namespace],

    def_map: DefMap,
    export_map2: ExportMap2,
    trait_map: TraitMap,
    external_exports: ExternalExports,
    last_private: LastPrivateMap,

    // Whether or not to print error messages. Can be set to true
    // when getting additional info for error message suggestions,
    // so as to avoid printing duplicate errors
    emit_errors: bool,

    used_imports: HashSet<NodeId>,
}

struct BuildReducedGraphVisitor<'a> {
    resolver: &'a mut Resolver,
}

impl<'a> Visitor<ReducedGraphParent> for BuildReducedGraphVisitor<'a> {

    fn visit_item(&mut self, item:@item, context:ReducedGraphParent) {
        let p = self.resolver.build_reduced_graph_for_item(item, context);
        visit::walk_item(self, item, p);
    }

    fn visit_foreign_item(&mut self, foreign_item: @foreign_item,
                          context:ReducedGraphParent) {
        self.resolver.build_reduced_graph_for_foreign_item(foreign_item,
                                                           context,
                                                           |r, c| {
            let mut v = BuildReducedGraphVisitor{ resolver: r };
            visit::walk_foreign_item(&mut v, foreign_item, c);
        })
    }

    fn visit_view_item(&mut self, view_item:&view_item, context:ReducedGraphParent) {
        self.resolver.build_reduced_graph_for_view_item(view_item, context);
    }

    fn visit_block(&mut self, block:P<Block>, context:ReducedGraphParent) {
        let np = self.resolver.build_reduced_graph_for_block(block, context);
        visit::walk_block(self, block, np);
    }

}

struct UnusedImportCheckVisitor<'a> { resolver: &'a Resolver }

impl<'a> Visitor<()> for UnusedImportCheckVisitor<'a> {
    fn visit_view_item(&mut self, vi:&view_item, _:()) {
        self.resolver.check_for_item_unused_imports(vi);
        visit::walk_view_item(self, vi, ());
    }
}

impl Resolver {
    /// The main name resolution procedure.
    fn resolve(&mut self, crate: &ast::Crate) {
        self.build_reduced_graph(crate);
        self.session.abort_if_errors();

        self.resolve_imports();
        self.session.abort_if_errors();

        self.record_exports();
        self.session.abort_if_errors();

        self.resolve_crate(crate);
        self.session.abort_if_errors();

        self.check_for_unused_imports(crate);
    }

    //
    // Reduced graph building
    //
    // Here we build the "reduced graph": the graph of the module tree without
    // any imports resolved.
    //

    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(&mut self, crate: &ast::Crate) {
        let initial_parent =
            ModuleReducedGraphParent(self.graph_root.get_module());

        let mut visitor = BuildReducedGraphVisitor { resolver: self, };
        visit::walk_crate(&mut visitor, crate, initial_parent);
    }

    /// Returns the current module tracked by the reduced graph parent.
    fn get_module_from_parent(&mut self,
                                  reduced_graph_parent: ReducedGraphParent)
                                  -> @Module {
        match reduced_graph_parent {
            ModuleReducedGraphParent(module_) => {
                return module_;
            }
        }
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
    fn add_child(&mut self,
                     name: Ident,
                     reduced_graph_parent: ReducedGraphParent,
                     duplicate_checking_mode: DuplicateCheckingMode,
                     // For printing errors
                     sp: Span)
                     -> (@NameBindings, ReducedGraphParent) {
        // If this is the immediate descendant of a module, then we add the
        // child name directly. Otherwise, we create or reuse an anonymous
        // module and add the child to that.

        let module_;
        match reduced_graph_parent {
            ModuleReducedGraphParent(parent_module) => {
                module_ = parent_module;
            }
        }

        // Add or reuse the child.
        let new_parent = ModuleReducedGraphParent(module_);
        let child_opt = {
            let children = module_.children.borrow();
            children.get().find_copy(&name.name)
        };
        match child_opt {
            None => {
                let child = @NameBindings();
                let mut children = module_.children.borrow_mut();
                children.get().insert(name.name, child);
                return (child, new_parent);
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
                        if (child.get_module_if_available().is_some()) {
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
                if (duplicate_type != NoError) {
                    // Return an error here by looking up the namespace that
                    // had the duplicate.
                    let ns = ns.unwrap();
                    self.resolve_error(sp,
                        format!("duplicate definition of {} `{}`",
                             namespace_error_to_str(duplicate_type),
                             self.session.str_of(name)));
                    {
                        let r = child.span_for_namespace(ns);
                        for sp in r.iter() {
                            self.session.span_note(*sp,
                                 format!("first definition of {} `{}` here",
                                      namespace_error_to_str(duplicate_type),
                                      self.session.str_of(name)));
                        }
                    }
                }
                return (child, new_parent);
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
                return ModuleParentLink(module_, name);
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(&mut self,
                                        item: @item,
                                        parent: ReducedGraphParent)
                                            -> ReducedGraphParent
    {
        let ident = item.ident;
        let sp = item.span;
        let is_public = item.vis == ast::public;

        match item.node {
            item_mod(..) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateModules, sp);

                let parent_link = self.get_parent_link(new_parent, ident);
                let def_id = DefId { crate: 0, node: item.id };
                name_bindings.define_module(parent_link,
                                            Some(def_id),
                                            NormalModuleKind,
                                            false,
                                            item.vis == ast::public,
                                            sp);

                ModuleReducedGraphParent(name_bindings.get_module())
            }

            item_foreign_mod(..) => parent,

            // These items live in the value namespace.
            item_static(_, m, _) => {
                let (name_bindings, _) =
                    self.add_child(ident, parent, ForbidDuplicateValues, sp);
                let mutbl = m == ast::MutMutable;

                name_bindings.define_value
                    (DefStatic(local_def(item.id), mutbl), sp, is_public);
                parent
            }
            item_fn(_, purity, _, _, _) => {
              let (name_bindings, new_parent) =
                self.add_child(ident, parent, ForbidDuplicateValues, sp);

                let def = DefFn(local_def(item.id), purity);
                name_bindings.define_value(def, sp, is_public);
                new_parent
            }

            // These items live in the type namespace.
            item_ty(..) => {
                let (name_bindings, _) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (DefTy(local_def(item.id)), sp, is_public);
                parent
            }

            item_enum(ref enum_definition, _) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (DefTy(local_def(item.id)), sp, is_public);

                for &variant in (*enum_definition).variants.iter() {
                    self.build_reduced_graph_for_variant(
                        variant,
                        local_def(item.id),
                        new_parent,
                        is_public);
                }
                parent
            }

            // These items live in both the type and value namespaces.
            item_struct(struct_def, _) => {
                // Adding to both Type and Value namespaces or just Type?
                let (forbid, ctor_id) = match struct_def.ctor_id {
                    Some(ctor_id)   => (ForbidDuplicateTypesAndValues, Some(ctor_id)),
                    None            => (ForbidDuplicateTypes, None)
                };

                let (name_bindings, new_parent) = self.add_child(ident, parent, forbid, sp);

                // Define a name in the type namespace.
                name_bindings.define_type(DefTy(local_def(item.id)), sp, is_public);

                // If this is a newtype or unit-like struct, define a name
                // in the value namespace as well
                ctor_id.while_some(|cid| {
                    name_bindings.define_value(DefStruct(local_def(cid)), sp,
                                               is_public);
                    None
                });

                // Record the def ID of this struct.
                self.structs.insert(local_def(item.id));

                new_parent
            }

            item_impl(_, None, ty, ref methods) => {
                // If this implements an anonymous trait, then add all the
                // methods within to a new module, if the type was defined
                // within this module.
                //
                // FIXME (#3785): This is quite unsatisfactory. Perhaps we
                // should modify anonymous traits to only be implementable in
                // the same module that declared the type.

                // Create the module and add all methods.
                match ty.node {
                    ty_path(ref path, _, _) if path.segments.len() == 1 => {
                        let name = path_to_ident(path);

                        let existing_parent_opt = {
                            let children = parent.children.borrow();
                            children.get().find_copy(&name.name)
                        };
                        let new_parent = match existing_parent_opt {
                            // It already exists
                            Some(child) if child.get_module_if_available()
                                                .is_some() &&
                                           child.get_module().kind.get() ==
                                                ImplModuleKind => {
                                ModuleReducedGraphParent(child.get_module())
                            }
                            // Create the module
                            _ => {
                                let (name_bindings, new_parent) =
                                    self.add_child(name,
                                                   parent,
                                                   ForbidDuplicateModules,
                                                   sp);

                                let parent_link =
                                    self.get_parent_link(new_parent, ident);
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
                            let (method_name_bindings, _) =
                                self.add_child(ident,
                                               new_parent,
                                               ForbidDuplicateValues,
                                               method.span);
                            let def = match method.explicit_self.node {
                                sty_static => {
                                    // Static methods become
                                    // `def_static_method`s.
                                    DefStaticMethod(local_def(method.id),
                                                      FromImpl(local_def(
                                                        item.id)),
                                                      method.purity)
                                }
                                _ => {
                                    // Non-static methods become
                                    // `def_method`s.
                                    DefMethod(local_def(method.id), None)
                                }
                            };

                            let is_public = method.vis == ast::public;
                            method_name_bindings.define_value(def,
                                                              method.span,
                                                              is_public);
                        }
                    }
                    _ => {}
                }

                parent
            }

            item_impl(_, Some(_), _, _) => parent,

            item_trait(_, _, ref methods) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                // Add all the methods within to a new module.
                let parent_link = self.get_parent_link(parent, ident);
                name_bindings.define_module(parent_link,
                                            Some(local_def(item.id)),
                                            TraitModuleKind,
                                            false,
                                            item.vis == ast::public,
                                            sp);
                let module_parent = ModuleReducedGraphParent(name_bindings.
                                                             get_module());

                // Add the names of all the methods to the trait info.
                let mut method_names = HashMap::new();
                for method in methods.iter() {
                    let ty_m = trait_method_to_ty_method(method);

                    let ident = ty_m.ident;

                    // Add it as a name in the trait module.
                    let def = match ty_m.explicit_self.node {
                        sty_static => {
                            // Static methods become `def_static_method`s.
                            DefStaticMethod(local_def(ty_m.id),
                                              FromTrait(local_def(item.id)),
                                              ty_m.purity)
                        }
                        _ => {
                            // Non-static methods become `def_method`s.
                            DefMethod(local_def(ty_m.id),
                                       Some(local_def(item.id)))
                        }
                    };

                    let (method_name_bindings, _) =
                        self.add_child(ident,
                                       module_parent,
                                       ForbidDuplicateValues,
                                       ty_m.span);
                    method_name_bindings.define_value(def, ty_m.span, true);

                    // Add it to the trait info if not static.
                    match ty_m.explicit_self.node {
                        sty_static => {}
                        _ => {
                            method_names.insert(ident.name, ());
                        }
                    }
                }

                let def_id = local_def(item.id);
                for (name, _) in method_names.iter() {
                    let mut method_map = self.method_map.borrow_mut();
                    if !method_map.get().contains_key(name) {
                        method_map.get().insert(*name, HashSet::new());
                    }
                    match method_map.get().find_mut(name) {
                        Some(s) => { s.insert(def_id); },
                        _ => fail!("Can't happen"),
                    }
                }

                name_bindings.define_type(DefTrait(def_id), sp, is_public);
                new_parent
            }

            item_mac(..) => {
                fail!("item macros unimplemented")
            }
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and/or value namespaces.
    fn build_reduced_graph_for_variant(&mut self,
                                       variant: &variant,
                                       item_id: DefId,
                                       parent: ReducedGraphParent,
                                       parent_public: bool) {
        let ident = variant.node.name;
        // XXX: this is unfortunate to have to do this privacy calculation
        //      here. This should be living in middle::privacy, but it's
        //      necessary to keep around in some form becaues of glob imports...
        let is_public = parent_public && variant.node.vis != ast::private;

        match variant.node.kind {
            tuple_variant_kind(_) => {
                let (child, _) = self.add_child(ident, parent, ForbidDuplicateValues,
                                                variant.span);
                child.define_value(DefVariant(item_id,
                                              local_def(variant.node.id), false),
                                   variant.span, is_public);
            }
            struct_variant_kind(_) => {
                let (child, _) = self.add_child(ident, parent, ForbidDuplicateTypesAndValues,
                                                variant.span);
                child.define_type(DefVariant(item_id,
                                             local_def(variant.node.id), true),
                                  variant.span, is_public);
                self.structs.insert(local_def(variant.node.id));
            }
        }
    }

    /// Constructs the reduced graph for one 'view item'. View items consist
    /// of imports and use directives.
    fn build_reduced_graph_for_view_item(&mut self,
                                             view_item: &view_item,
                                             parent: ReducedGraphParent) {
        match view_item.node {
            view_item_use(ref view_paths) => {
                for view_path in view_paths.iter() {
                    // Extract and intern the module part of the path. For
                    // globs and lists, the path is found directly in the AST;
                    // for simple paths we have to munge the path a little.

                    let mut module_path = ~[];
                    match view_path.node {
                        view_path_simple(_, ref full_path, _) => {
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

                        view_path_glob(ref module_ident_path, _) |
                        view_path_list(ref module_ident_path, _, _) => {
                            for segment in module_ident_path.segments.iter() {
                                module_path.push(segment.identifier)
                            }
                        }
                    }

                    // Build up the import directives.
                    let module_ = self.get_module_from_parent(parent);
                    let is_public = view_item.vis == ast::public;
                    match view_path.node {
                        view_path_simple(binding, ref full_path, id) => {
                            let source_ident =
                                full_path.segments.last().identifier;
                            let subclass = @SingleImport(binding,
                                                         source_ident);
                            self.build_import_directive(module_,
                                                        module_path,
                                                        subclass,
                                                        view_path.span,
                                                        id,
                                                        is_public);
                        }
                        view_path_list(_, ref source_idents, _) => {
                            for source_ident in source_idents.iter() {
                                let name = source_ident.node.name;
                                let subclass = @SingleImport(name, name);
                                self.build_import_directive(
                                    module_,
                                    module_path.clone(),
                                    subclass,
                                    source_ident.span,
                                    source_ident.node.id,
                                    is_public);
                            }
                        }
                        view_path_glob(_, id) => {
                            self.build_import_directive(module_,
                                                        module_path,
                                                        @GlobImport,
                                                        view_path.span,
                                                        id,
                                                        is_public);
                        }
                    }
                }
            }

            view_item_extern_mod(name, _, _, node_id) => {
                // n.b. we don't need to look at the path option here, because cstore already did
                match self.session.cstore.find_extern_mod_stmt_cnum(node_id) {
                    Some(crate_id) => {
                        let def_id = DefId { crate: crate_id, node: 0 };
                        self.external_exports.insert(def_id);
                        let parent_link = ModuleParentLink
                            (self.get_module_from_parent(parent), name);
                        let external_module = @Module::new(parent_link,
                                                          Some(def_id),
                                                          NormalModuleKind,
                                                          false,
                                                          true);

                        {
                            let mut external_module_children =
                                parent.external_module_children.borrow_mut();
                            external_module_children.get().insert(
                                name.name,
                                external_module);
                        }

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
                                            foreign_item: @foreign_item,
                                            parent: ReducedGraphParent,
                                            f: |&mut Resolver,
                                                ReducedGraphParent|) {
        let name = foreign_item.ident;
        let is_public = foreign_item.vis == ast::public;
        let (name_bindings, new_parent) =
            self.add_child(name, parent, ForbidDuplicateValues,
                           foreign_item.span);

        match foreign_item.node {
            foreign_item_fn(_, ref generics) => {
                let def = DefFn(local_def(foreign_item.id), unsafe_fn);
                name_bindings.define_value(def, foreign_item.span, is_public);

                self.with_type_parameter_rib(
                    HasTypeParameters(generics,
                                      foreign_item.id,
                                      0,
                                      NormalRibKind),
                    |this| f(this, new_parent));
            }
            foreign_item_static(_, m) => {
                let def = DefStatic(local_def(foreign_item.id), m);
                name_bindings.define_value(def, foreign_item.span, is_public);

                f(self, new_parent)
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

            let parent_module = self.get_module_from_parent(parent);
            let new_module = @Module::new(
                BlockParentLink(parent_module, block_id),
                None,
                AnonymousModuleKind,
                false,
                false);
            {
                let mut anonymous_children = parent_module.anonymous_children
                                                          .borrow_mut();
                anonymous_children.get().insert(block_id, new_module);
                ModuleReducedGraphParent(new_module)
            }
        } else {
            parent
        }
    }

    fn handle_external_def(&mut self,
                           def: Def,
                           vis: visibility,
                           child_name_bindings: @NameBindings,
                           final_ident: &str,
                           ident: Ident,
                           new_parent: ReducedGraphParent) {
        debug!("(building reduced graph for \
                external crate) building external def, priv {:?}",
               vis);
        let is_public = vis == ast::public;
        let is_exported = is_public && match new_parent {
            ModuleReducedGraphParent(module) => {
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
            match child_name_bindings.type_def.get() {
              Some(TypeNsDef { module_def: Some(module_def), .. }) => {
                debug!("(building reduced graph for external crate) \
                        already created module");
                module_def.def_id.set(Some(def_id));
              }
              Some(_) | None => {
                debug!("(building reduced graph for \
                        external crate) building module \
                        {}", final_ident);
                let parent_link = self.get_parent_link(new_parent, ident);

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
          DefVariant(_, variant_id, is_struct) => {
            debug!("(building reduced graph for external crate) building \
                    variant {}",
                   final_ident);
            // We assume the parent is visible, or else we wouldn't have seen
            // it. Also variants are public-by-default if the parent was also
            // public.
            let is_public = vis != ast::private;
            if is_struct {
                child_name_bindings.define_type(def, DUMMY_SP, is_public);
                self.structs.insert(variant_id);
            } else {
                child_name_bindings.define_value(def, DUMMY_SP, is_public);
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
                csearch::get_trait_method_def_ids(self.session.cstore, def_id);
              let mut interned_method_names = HashSet::new();
              for &method_def_id in method_def_ids.iter() {
                  let (method_name, explicit_self) =
                      csearch::get_method_name_and_explicit_self(self.session.cstore,
                                                                 method_def_id);

                  debug!("(building reduced graph for \
                          external crate) ... adding \
                          trait method '{}'",
                         self.session.str_of(method_name));

                  // Add it to the trait info if not static.
                  if explicit_self != sty_static {
                      interned_method_names.insert(method_name.name);
                  }
                  if is_exported {
                      self.external_exports.insert(method_def_id);
                  }
              }
              for name in interned_method_names.iter() {
                  let mut method_map = self.method_map.borrow_mut();
                  if !method_map.get().contains_key(name) {
                      method_map.get().insert(*name, HashSet::new());
                  }
                  match method_map.get().find_mut(name) {
                      Some(s) => { s.insert(def_id); },
                      _ => fail!("Can't happen"),
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
            if csearch::get_struct_fields(self.session.cstore, def_id).len() == 0 {
                child_name_bindings.define_value(def, DUMMY_SP, is_public);
            }
            self.structs.insert(def_id);
          }
          DefMethod(..) => {
              debug!("(building reduced graph for external crate) \
                      ignoring {:?}", def);
              // Ignored; handled elsewhere.
          }
          DefSelf(..) | DefArg(..) | DefLocal(..) |
          DefPrimTy(..) | DefTyParam(..) | DefBinding(..) |
          DefUse(..) | DefUpvar(..) | DefRegion(..) |
          DefTyParamBinder(..) | DefLabel(..) | DefSelfTy(..) => {
            fail!("didn't expect `{:?}`", def);
          }
        }
    }

    /// Builds the reduced graph for a single item in an external crate.
    fn build_reduced_graph_for_external_crate_def(&mut self,
                                                  root: @Module,
                                                  def_like: DefLike,
                                                  ident: Ident,
                                                  visibility: visibility) {
        match def_like {
            DlDef(def) => {
                // Add the new child item, if necessary.
                match def {
                    DefForeignMod(def_id) => {
                        // Foreign modules have no names. Recur and populate
                        // eagerly.
                        csearch::each_child_of_item(self.session.cstore,
                                                    def_id,
                                                    |def_like,
                                                     child_ident,
                                                     vis| {
                            self.build_reduced_graph_for_external_crate_def(
                                root,
                                def_like,
                                child_ident,
                                vis)
                        });
                    }
                    _ => {
                        let (child_name_bindings, new_parent) =
                            self.add_child(ident,
                                           ModuleReducedGraphParent(root),
                                           OverwriteDuplicates,
                                           DUMMY_SP);

                        self.handle_external_def(def,
                                                 visibility,
                                                 child_name_bindings,
                                                 self.session.str_of(ident),
                                                 ident,
                                                 new_parent);
                    }
                }
            }
            DlImpl(def) => {
                // We only process static methods of impls here.
                match csearch::get_type_name_if_impl(self.session.cstore, def) {
                    None => {}
                    Some(final_ident) => {
                        let static_methods_opt =
                            csearch::get_static_methods_if_impl(self.session.cstore, def);
                        match static_methods_opt {
                            Some(ref static_methods) if
                                static_methods.len() >= 1 => {
                                debug!("(building reduced graph for \
                                        external crate) processing \
                                        static methods for type name {}",
                                        self.session.str_of(
                                            final_ident));

                                let (child_name_bindings, new_parent) =
                                    self.add_child(
                                        final_ident,
                                        ModuleReducedGraphParent(root),
                                        OverwriteDuplicates,
                                        DUMMY_SP);

                                // Process the static methods. First,
                                // create the module.
                                let type_module;
                                match child_name_bindings.type_def.get() {
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
                                            self.get_parent_link(new_parent,
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
                                           self.session.str_of(ident));

                                    let (method_name_bindings, _) =
                                        self.add_child(ident,
                                                       new_parent,
                                                       OverwriteDuplicates,
                                                       DUMMY_SP);
                                    let def = DefFn(
                                        static_method_info.def_id,
                                        static_method_info.purity);

                                    method_name_bindings.define_value(
                                        def, DUMMY_SP,
                                        visibility == ast::public);
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
    fn populate_external_module(&mut self, module: @Module) {
        debug!("(populating external module) attempting to populate {}",
               self.module_to_str(module));

        let def_id = match module.def_id.get() {
            None => {
                debug!("(populating external module) ... no def ID!");
                return
            }
            Some(def_id) => def_id,
        };

        csearch::each_child_of_item(self.session.cstore,
                                    def_id,
                                    |def_like, child_ident, visibility| {
            debug!("(populating external module) ... found ident: {}",
                   token::ident_to_str(&child_ident));
            self.build_reduced_graph_for_external_crate_def(module,
                                                            def_like,
                                                            child_ident,
                                                            visibility)
        });
        module.populated.set(true)
    }

    /// Ensures that the reduced graph rooted at the given external module
    /// is built, building it if it is not.
    fn populate_module_if_necessary(&mut self, module: @Module) {
        if !module.populated.get() {
            self.populate_external_module(module)
        }
        assert!(module.populated.get())
    }

    /// Builds the reduced graph rooted at the 'use' directive for an external
    /// crate.
    fn build_reduced_graph_for_external_crate(&mut self,
                                              root: @Module) {
        csearch::each_top_level_item_of_crate(self.session.cstore,
                                              root.def_id
                                                  .get()
                                                  .unwrap()
                                                  .crate,
                                              |def_like, ident, visibility| {
            self.build_reduced_graph_for_external_crate_def(root,
                                                            def_like,
                                                            ident,
                                                            visibility)
        });
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(&mut self,
                              module_: @Module,
                              module_path: ~[Ident],
                              subclass: @ImportDirectiveSubclass,
                              span: Span,
                              id: NodeId,
                              is_public: bool) {
        let directive = @ImportDirective::new(module_path,
                                              subclass, span, id,
                                              is_public);

        {
            let mut imports = module_.imports.borrow_mut();
            imports.get().push(directive);
        }

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match *subclass {
            SingleImport(target, _) => {
                debug!("(building import directive) building import \
                        directive: {}::{}",
                       self.idents_to_str(directive.module_path),
                       self.session.str_of(target));

                let mut import_resolutions = module_.import_resolutions
                                                    .borrow_mut();
                match import_resolutions.get().find(&target.name) {
                    Some(&resolution) => {
                        debug!("(building import directive) bumping \
                                reference");
                        resolution.outstanding_references.set(
                            resolution.outstanding_references.get() + 1);

                        // the source of this name is different now
                        resolution.type_id.set(id);
                        resolution.value_id.set(id);
                    }
                    None => {
                        debug!("(building import directive) creating new");
                        let resolution = @ImportResolution::new(id, is_public);
                        resolution.outstanding_references.set(1);
                        import_resolutions.get().insert(target.name,
                                                        resolution);
                    }
                }
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.glob_count.set(module_.glob_count.get() + 1);
            }
        }

        self.unresolved_imports += 1;
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
            self.resolve_imports_for_module_subtree(module_root);

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
    fn resolve_imports_for_module_subtree(&mut self,
                                              module_: @Module) {
        debug!("(resolving imports for module subtree) resolving {}",
               self.module_to_str(module_));
        self.resolve_imports_for_module(module_);

        self.populate_module_if_necessary(module_);
        {
            let children = module_.children.borrow();
            for (_, &child_node) in children.get().iter() {
                match child_node.get_module_if_available() {
                    None => {
                        // Nothing to do.
                    }
                    Some(child_module) => {
                        self.resolve_imports_for_module_subtree(child_module);
                    }
                }
            }
        }

        let anonymous_children = module_.anonymous_children.borrow();
        for (_, &child_module) in anonymous_children.get().iter() {
            self.resolve_imports_for_module_subtree(child_module);
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(&mut self, module: @Module) {
        if module.all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for \
                   {}",
                   self.module_to_str(module));
            return;
        }

        let mut imports = module.imports.borrow_mut();
        let import_count = imports.get().len();
        while module.resolved_import_count.get() < import_count {
            let import_index = module.resolved_import_count.get();
            let import_directive = imports.get()[import_index];
            match self.resolve_import_for_module(module, import_directive) {
                Failed => {
                    // We presumably emitted an error. Continue.
                    let msg = format!("failed to resolve import `{}`",
                                   self.import_path_to_str(
                                       import_directive.module_path,
                                       *import_directive.subclass));
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

    fn idents_to_str(&mut self, idents: &[Ident]) -> ~str {
        let mut first = true;
        let mut result = ~"";
        for ident in idents.iter() {
            if first {
                first = false
            } else {
                result.push_str("::")
            }
            result.push_str(self.session.str_of(*ident));
        };
        return result;
    }

    fn path_idents_to_str(&mut self, path: &Path) -> ~str {
        let identifiers: ~[ast::Ident] = path.segments
                                             .iter()
                                             .map(|seg| seg.identifier)
                                             .collect();
        self.idents_to_str(identifiers)
    }

    fn import_directive_subclass_to_str(&mut self,
                                            subclass: ImportDirectiveSubclass)
                                            -> @str {
        match subclass {
            SingleImport(_target, source) => self.session.str_of(source),
            GlobImport => @"*"
        }
    }

    fn import_path_to_str(&mut self,
                              idents: &[Ident],
                              subclass: ImportDirectiveSubclass)
                              -> @str {
        if idents.is_empty() {
            self.import_directive_subclass_to_str(subclass)
        } else {
            (format!("{}::{}",
                  self.idents_to_str(idents),
                  self.import_directive_subclass_to_str(subclass))).to_managed()
        }
    }

    /// Attempts to resolve the given import. The return value indicates
    /// failure if we're certain the name does not exist, indeterminate if we
    /// don't know whether the name exists at the moment due to other
    /// currently-unresolved imports, or success if we know the name exists.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import_for_module(&mut self,
                                 module_: @Module,
                                 import_directive: @ImportDirective)
                                 -> ResolveResult<()> {
        let mut resolution_result = Failed;
        let module_path = &import_directive.module_path;

        debug!("(resolving import for module) resolving import `{}::...` in \
                `{}`",
               self.idents_to_str(*module_path),
               self.module_to_str(module_));

        // First, resolve the module path for the directive, if necessary.
        let container = if module_path.len() == 0 {
            // Use the crate root.
            Some((self.graph_root.get_module(), AllPublic))
        } else {
            match self.resolve_module_path(module_,
                                           *module_path,
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

                match *import_directive.subclass {
                    SingleImport(target, source) => {
                        resolution_result =
                            self.resolve_single_import(module_,
                                                       containing_module,
                                                       target,
                                                       source,
                                                       import_directive,
                                                       lp);
                    }
                    GlobImport => {
                        resolution_result =
                            self.resolve_glob_import(module_,
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
            match *import_directive.subclass {
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

    fn create_name_bindings_from_module(module: @Module) -> NameBindings {
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
                             module_: @Module,
                             containing_module: @Module,
                             target: Ident,
                             source: Ident,
                             directive: &ImportDirective,
                             lp: LastPrivate)
                                 -> ResolveResult<()> {
        debug!("(resolving single import) resolving `{}` = `{}::{}` from \
                `{}` id {}, last private {:?}",
               self.session.str_of(target),
               self.module_to_str(containing_module),
               self.session.str_of(source),
               self.module_to_str(module_),
               directive.id,
               lp);

        // We need to resolve both namespaces for this to succeed.
        //

        let mut value_result = UnknownResult;
        let mut type_result = UnknownResult;

        // Search for direct children of the containing module.
        self.populate_module_if_necessary(containing_module);

        {
            let children = containing_module.children.borrow();
            match children.get().find(&source.name) {
                None => {
                    // Continue.
                }
                Some(child_name_bindings) => {
                    if child_name_bindings.defined_in_namespace(ValueNS) {
                        value_result = BoundResult(containing_module,
                                                   *child_name_bindings);
                    }
                    if child_name_bindings.defined_in_namespace(TypeNS) {
                        type_result = BoundResult(containing_module,
                                                  *child_name_bindings);
                    }
                }
            }
        }

        // Unless we managed to find a result in both namespaces (unlikely),
        // search imports as well.
        let mut used_reexport = false;
        match (value_result, type_result) {
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

                // Now search the exported imports within the containing
                // module.

                let import_resolutions = containing_module.import_resolutions
                                                          .borrow();
                match import_resolutions.get().find(&source.name) {
                    None => {
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
                            if import_resolution.outstanding_references.get()
                                == 0 => {

                        fn get_binding(this: &mut Resolver,
                                       import_resolution: @ImportResolution,
                                       namespace: Namespace)
                                    -> NamespaceResult {

                            // Import resolutions must be declared with "pub"
                            // in order to be exported.
                            if !import_resolution.is_public.get() {
                                return UnboundResult;
                            }

                            match (*import_resolution).
                                    target_for_namespace(namespace) {
                                None => {
                                    return UnboundResult;
                                }
                                Some(target) => {
                                    let id = import_resolution.id(namespace);
                                    this.used_imports.insert(id);
                                    return BoundResult(target.target_module,
                                                       target.bindings);
                                }
                            }
                        }

                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.
                        if value_result.is_unknown() {
                            value_result = get_binding(self, *import_resolution,
                                                       ValueNS);
                            used_reexport = import_resolution.is_public.get();
                        }
                        if type_result.is_unknown() {
                            type_result = get_binding(self, *import_resolution,
                                                      TypeNS);
                            used_reexport = import_resolution.is_public.get();
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
        let mut used_public = false;
        match type_result {
            BoundResult(..) => {}
            _ => {
                let module_opt = {
                    let mut external_module_children =
                        containing_module.external_module_children
                                         .borrow_mut();
                    external_module_children.get().find_copy(&source.name)
                };
                match module_opt {
                    None => {} // Continue.
                    Some(module) => {
                        let name_bindings =
                            @Resolver::create_name_bindings_from_module(
                                module);
                        type_result = BoundResult(containing_module,
                                                  name_bindings);
                        used_public = true;
                    }
                }
            }
        }

        // We've successfully resolved the import. Write the results in.
        let import_resolution = {
            let import_resolutions = module_.import_resolutions.borrow();
            assert!(import_resolutions.get().contains_key(&target.name));
            import_resolutions.get().get_copy(&target.name)
        };

        match value_result {
            BoundResult(target_module, name_bindings) => {
                debug!("(resolving single import) found value target");
                import_resolution.value_target.set(
                    Some(Target::new(target_module, name_bindings)));
                import_resolution.value_id.set(directive.id);
                used_public = name_bindings.defined_in_public_namespace(ValueNS);
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!("value result should be known at this point");
            }
        }
        match type_result {
            BoundResult(target_module, name_bindings) => {
                debug!("(resolving single import) found type target: {:?}",
                        name_bindings.type_def.get().unwrap().type_def);
                import_resolution.type_target.set(
                    Some(Target::new(target_module, name_bindings)));
                import_resolution.type_id.set(directive.id);
                used_public = name_bindings.defined_in_public_namespace(TypeNS);
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!("type result should be known at this point");
            }
        }

        if import_resolution.value_target.get().is_none() &&
           import_resolution.type_target.get().is_none() {
            let msg = format!("unresolved import: there is no \
                               `{}` in `{}`",
                              self.session.str_of(source),
                              self.module_to_str(containing_module));
            self.resolve_error(directive.span, msg);
            return Failed;
        }
        let used_public = used_reexport || used_public;

        assert!(import_resolution.outstanding_references.get() >= 1);
        import_resolution.outstanding_references.set(
            import_resolution.outstanding_references.get() - 1);

        // record what this import resolves to for later uses in documentation,
        // this may resolve to either a value or a type, but for documentation
        // purposes it's good enough to just favor one over the other.
        match import_resolution.value_target.get() {
            Some(target) => {
                let def = target.bindings.def_for_namespace(ValueNS).unwrap();
                let mut def_map = self.def_map.borrow_mut();
                def_map.get().insert(directive.id, def);
                let did = def_id_of_def(def);
                self.last_private.insert(directive.id,
                    if used_public {lp} else {DependsOn(did)});
            }
            None => {}
        }
        match import_resolution.type_target.get() {
            Some(target) => {
                let def = target.bindings.def_for_namespace(TypeNS).unwrap();
                let mut def_map = self.def_map.borrow_mut();
                def_map.get().insert(directive.id, def);
                let did = def_id_of_def(def);
                self.last_private.insert(directive.id,
                    if used_public {lp} else {DependsOn(did)});
            }
            None => {}
        }

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    // Resolves a glob import. Note that this function cannot fail; it either
    // succeeds or bails out (as importing * from an empty module or a module
    // that exports nothing is valid).
    fn resolve_glob_import(&mut self,
                           module_: @Module,
                           containing_module: @Module,
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
        for (ident, target_import_resolution) in import_resolutions.get()
                                                                   .iter() {
            debug!("(resolving glob import) writing module resolution \
                    {:?} into `{}`",
                   target_import_resolution.type_target.get().is_none(),
                   self.module_to_str(module_));

            if !target_import_resolution.is_public.get() {
                debug!("(resolving glob import) nevermind, just kidding");
                continue
            }

            // Here we merge two import resolutions.
            let mut import_resolutions = module_.import_resolutions
                                                .borrow_mut();
            match import_resolutions.get().find(ident) {
                None => {
                    // Simple: just copy the old import resolution.
                    let new_import_resolution =
                        @ImportResolution::new(id, is_public);
                    new_import_resolution.value_target.set(
                        target_import_resolution.value_target.get());
                    new_import_resolution.type_target.set(
                        target_import_resolution.type_target.get());

                    import_resolutions.get().insert
                        (*ident, new_import_resolution);
                }
                Some(&dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match target_import_resolution.value_target.get() {
                        None => {
                            // Continue.
                        }
                        Some(value_target) => {
                            dest_import_resolution.value_target.set(
                                Some(value_target));
                        }
                    }
                    match target_import_resolution.type_target.get() {
                        None => {
                            // Continue.
                        }
                        Some(type_target) => {
                            dest_import_resolution.type_target.set(
                                Some(type_target));
                        }
                    }
                    dest_import_resolution.is_public.set(is_public);
                }
            }
        }

        let merge_import_resolution = |name, name_bindings: @NameBindings| {
            let dest_import_resolution;
            let mut import_resolutions = module_.import_resolutions
                                                .borrow_mut();
            match import_resolutions.get().find(&name) {
                None => {
                    // Create a new import resolution from this child.
                    dest_import_resolution =
                        @ImportResolution::new(id, is_public);
                    import_resolutions.get().insert(name,
                                                    dest_import_resolution);
                }
                Some(&existing_import_resolution) => {
                    dest_import_resolution = existing_import_resolution;
                }
            }

            debug!("(resolving glob import) writing resolution `{}` in `{}` \
                    to `{}`",
                   interner_get(name),
                   self.module_to_str(containing_module),
                   self.module_to_str(module_));

            // Merge the child item into the import resolution.
            if name_bindings.defined_in_public_namespace(ValueNS) {
                debug!("(resolving glob import) ... for value target");
                dest_import_resolution.value_target.set(
                    Some(Target::new(containing_module, name_bindings)));
                dest_import_resolution.value_id.set(id);
            }
            if name_bindings.defined_in_public_namespace(TypeNS) {
                debug!("(resolving glob import) ... for type target");
                dest_import_resolution.type_target.set(
                    Some(Target::new(containing_module, name_bindings)));
                dest_import_resolution.type_id.set(id);
            }
            dest_import_resolution.is_public.set(is_public);
        };

        // Add all children from the containing module.
        self.populate_module_if_necessary(containing_module);

        {
            let children = containing_module.children.borrow();
            for (&name, name_bindings) in children.get().iter() {
                merge_import_resolution(name, *name_bindings);
            }
        }

        // Add external module children from the containing module.
        {
            let external_module_children =
                containing_module.external_module_children.borrow();
            for (&name, module) in external_module_children.get().iter() {
                let name_bindings =
                    @Resolver::create_name_bindings_from_module(*module);
                merge_import_resolution(name, name_bindings);
            }
        }

        // Record the destination of this import
        match containing_module.def_id.get() {
            Some(did) => {
                let mut def_map = self.def_map.borrow_mut();
                def_map.get().insert(id, DefMod(did));
                self.last_private.insert(id, lp);
            }
            None => {}
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(&mut self,
                                     module_: @Module,
                                     module_path: &[Ident],
                                     index: uint,
                                     span: Span,
                                     name_search_type: NameSearchType,
                                     lp: LastPrivate)
                                -> ResolveResult<(@Module, LastPrivate)> {
        let mut search_module = module_;
        let mut index = index;
        let module_path_len = module_path.len();
        let mut closest_private = lp;

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.
        while index < module_path_len {
            let name = module_path[index];
            match self.resolve_name_in_module(search_module,
                                              name,
                                              TypeNS,
                                              name_search_type) {
                Failed => {
                    let segment_name = self.session.str_of(name);
                    let module_name = self.module_to_str(search_module);
                    if "???" == module_name {
                        let span = Span {
                            lo: span.lo,
                            hi: span.lo + Pos::from_uint(segment_name.len()),
                            expn_info: span.expn_info,
                        };
                        self.resolve_error(span,
                                              format!("unresolved import. maybe \
                                                    a missing `extern mod \
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
                            self.session.str_of(name));
                    return Indeterminate;
                }
                Success((target, used_proxy)) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    match target.bindings.type_def.get() {
                        Some(type_def) => {
                            match type_def.module_def {
                                None => {
                                    // Not a module.
                                    self.resolve_error(span,
                                                          format!("not a \
                                                                module `{}`",
                                                               self.session.
                                                                   str_of(
                                                                    name)));
                                    return Failed;
                                }
                                Some(module_def) => {
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
                                            search_module = module_def;

                                            // Keep track of the closest
                                            // private module used when
                                            // resolving this import chain.
                                            if !used_proxy &&
                                               !search_module.is_public {
                                                match search_module.def_id
                                                                   .get() {
                                                    Some(did) => {
                                                        closest_private =
                                                            DependsOn(did);
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
                                                       self.session.str_of(
                                                            name)));
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
                           module_: @Module,
                           module_path: &[Ident],
                           use_lexical_scope: UseLexicalScopeFlag,
                           span: Span,
                           name_search_type: NameSearchType)
                               -> ResolveResult<(@Module, LastPrivate)> {
        let module_path_len = module_path.len();
        assert!(module_path_len > 0);

        debug!("(resolving module path for import) processing `{}` rooted at \
               `{}`",
               self.idents_to_str(module_path),
               self.module_to_str(module_));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_,
                                                              module_path);

        let search_module;
        let start_index;
        let last_private;
        match module_prefix_result {
            Failed => {
                let mpath = self.idents_to_str(module_path);
                match mpath.rfind(':') {
                    Some(idx) => {
                        self.resolve_error(span, format!("unresolved import: could not find `{}` \
                                                         in `{}`",
                                                         // idx +- 1 to account for the colons
                                                         // on either side
                                                         mpath.slice_from(idx + 1),
                                                         mpath.slice_to(idx - 1)));
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
                        last_private = AllPublic;
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
                                last_private = AllPublic;
                            }
                        }
                    }
                }
            }
            Success(PrefixFound(containing_module, index)) => {
                search_module = containing_module;
                start_index = index;
                last_private = DependsOn(containing_module.def_id
                                                          .get()
                                                          .unwrap());
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
                                     module_: @Module,
                                     name: Ident,
                                     namespace: Namespace,
                                     search_through_modules:
                                     SearchThroughModulesFlag)
                                    -> ResolveResult<(Target, bool)> {
        debug!("(resolving item in lexical scope) resolving `{}` in \
                namespace {:?} in `{}`",
               self.session.str_of(name),
               namespace,
               self.module_to_str(module_));

        // The current module node is handled specially. First, check for
        // its immediate children.
        self.populate_module_if_necessary(module_);

        {
            let children = module_.children.borrow();
            match children.get().find(&name.name) {
                Some(name_bindings)
                        if name_bindings.defined_in_namespace(namespace) => {
                    debug!("top name bindings succeeded");
                    return Success((Target::new(module_, *name_bindings),
                                   false));
                }
                Some(_) | None => { /* Not found; continue. */ }
            }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.
        let import_resolutions = module_.import_resolutions.borrow();
        match import_resolutions.get().find(&name.name) {
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
                        self.used_imports.insert(import_resolution.id(namespace));
                        return Success((target, false));
                    }
                }
            }
        }

        // Search for external modules.
        if namespace == TypeNS {
            let module_opt = {
                let external_module_children =
                    module_.external_module_children.borrow();
                external_module_children.get().find_copy(&name.name)
            };
            match module_opt {
                None => {}
                Some(module) => {
                    let name_bindings =
                        @Resolver::create_name_bindings_from_module(module);
                    debug!("lower name bindings succeeded");
                    return Success((Target::new(module_, name_bindings), false));
                }
            }
        }

        // Finally, proceed up the scope chain looking for parent modules.
        let mut search_module = module_;
        loop {
            // Go to the next parent.
            match search_module.parent_link {
                NoParentLink => {
                    // No more parents. This module was unresolved.
                    debug!("(resolving item in lexical scope) unresolved \
                            module");
                    return Failed;
                }
                ModuleParentLink(parent_module_node, _) => {
                    match search_through_modules {
                        DontSearchThroughModules => {
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
                                    search_module = parent_module_node;
                                }
                            }
                        }
                        SearchThroughModules => {
                            search_module = parent_module_node;
                        }
                    }
                }
                BlockParentLink(parent_module_node, _) => {
                    search_module = parent_module_node;
                }
            }

            // Resolve the name in the parent module.
            match self.resolve_name_in_module(search_module,
                                              name,
                                              namespace,
                                              PathSearch) {
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
                                       module_: @Module,
                                       name: Ident)
                                -> ResolveResult<@Module> {
        // If this module is an anonymous module, resolve the item in the
        // lexical scope. Otherwise, resolve the item from the crate root.
        let resolve_result = self.resolve_item_in_lexical_scope(
            module_, name, TypeNS, DontSearchThroughModules);
        match resolve_result {
            Success((target, _)) => {
                let bindings = &*target.bindings;
                match bindings.type_def.get() {
                    Some(type_def) => {
                        match type_def.module_def {
                            None => {
                                error!("!!! (resolving module in lexical \
                                        scope) module wasn't actually a \
                                        module!");
                                return Failed;
                            }
                            Some(module_def) => {
                                return Success(module_def);
                            }
                        }
                    }
                    None => {
                        error!("!!! (resolving module in lexical scope) module
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
    fn get_nearest_normal_module_parent(&mut self, module_: @Module)
                                            -> Option<@Module> {
        let mut module_ = module_;
        loop {
            match module_.parent_link {
                NoParentLink => return None,
                ModuleParentLink(new_module, _) |
                BlockParentLink(new_module, _) => {
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
    fn get_nearest_normal_module_parent_or_self(&mut self, module_: @Module)
                                                -> @Module {
        match module_.kind.get() {
            NormalModuleKind => return module_,
            ExternModuleKind |
            TraitModuleKind |
            ImplModuleKind |
            AnonymousModuleKind => {
                match self.get_nearest_normal_module_parent(module_) {
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
                             module_: @Module,
                             module_path: &[Ident])
                                 -> ResolveResult<ModulePrefixResult> {
        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut containing_module;
        let mut i;
        if "self" == token::ident_to_str(&module_path[0]) {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 1;
        } else if "super" == token::ident_to_str(&module_path[0]) {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 0;  // We'll handle `super` below.
        } else {
            return Success(NoPrefixFound);
        }

        // Now loop through all the `super`s we find.
        while i < module_path.len() &&
                "super" == token::ident_to_str(&module_path[i]) {
            debug!("(resolving module prefix) resolving `super` at {}",
                   self.module_to_str(containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => return Failed,
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at {}",
               self.module_to_str(containing_module));

        return Success(PrefixFound(containing_module, i));
    }

    /// Attempts to resolve the supplied name in the given module for the
    /// given namespace. If successful, returns the target corresponding to
    /// the name.
    ///
    /// The boolean returned on success is an indicator of whether this lookup
    /// passed through a public re-export proxy.
    fn resolve_name_in_module(&mut self,
                              module_: @Module,
                              name: Ident,
                              namespace: Namespace,
                              name_search_type: NameSearchType)
                              -> ResolveResult<(Target, bool)> {
        debug!("(resolving name in module) resolving `{}` in `{}`",
               self.session.str_of(name),
               self.module_to_str(module_));

        // First, check the direct children of the module.
        self.populate_module_if_necessary(module_);

        {
            let children = module_.children.borrow();
            match children.get().find(&name.name) {
                Some(name_bindings)
                        if name_bindings.defined_in_namespace(namespace) => {
                    debug!("(resolving name in module) found node as child");
                    return Success((Target::new(module_, *name_bindings),
                                   false));
                }
                Some(_) | None => {
                    // Continue.
                }
            }
        }

        // Next, check the module's imports if necessary.

        // If this is a search of all imports, we should be done with glob
        // resolution at this point.
        if name_search_type == PathSearch {
            assert_eq!(module_.glob_count.get(), 0);
        }

        // Check the list of resolved imports.
        let import_resolutions = module_.import_resolutions.borrow();
        match import_resolutions.get().find(&name.name) {
            Some(import_resolution) => {
                if import_resolution.is_public.get() &&
                        import_resolution.outstanding_references.get() != 0 {
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
                        self.used_imports.insert(import_resolution.id(namespace));
                        return Success((target, true));
                    }
                }
            }
            None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            let module_opt = {
                let external_module_children =
                    module_.external_module_children.borrow();
                external_module_children.get().find_copy(&name.name)
            };
            match module_opt {
                None => {}
                Some(module) => {
                    let name_bindings =
                        @Resolver::create_name_bindings_from_module(module);
                    return Success((Target::new(module_, name_bindings), false));
                }
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve `{}`",
               self.session.str_of(name));
        return Failed;
    }

    fn report_unresolved_imports(&mut self, module_: @Module) {
        let index = module_.resolved_import_count.get();
        let mut imports = module_.imports.borrow_mut();
        let import_count = imports.get().len();
        if index != import_count {
            let sn = self.session
                         .codemap
                         .span_to_snippet(imports.get()[index].span)
                         .unwrap();
            if sn.contains("::") {
                self.resolve_error(imports.get()[index].span,
                                   "unresolved import");
            } else {
                let err = format!("unresolved import (maybe you meant `{}::*`?)",
                               sn.slice(0, sn.len()));
                self.resolve_error(imports.get()[index].span, err);
            }
        }

        // Descend into children and anonymous children.
        self.populate_module_if_necessary(module_);

        {
            let children = module_.children.borrow();
            for (_, &child_node) in children.get().iter() {
                match child_node.get_module_if_available() {
                    None => {
                        // Continue.
                    }
                    Some(child_module) => {
                        self.report_unresolved_imports(child_module);
                    }
                }
            }
        }

        let anonymous_children = module_.anonymous_children.borrow();
        for (_, &module_) in anonymous_children.get().iter() {
            self.report_unresolved_imports(module_);
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
                                             module_: @Module) {
        // If this isn't a local crate, then bail out. We don't need to record
        // exports for nonlocal crates.

        match module_.def_id.get() {
            Some(def_id) if def_id.crate == LOCAL_CRATE => {
                // OK. Continue.
                debug!("(recording exports for module subtree) recording \
                        exports for local module `{}`",
                       self.module_to_str(module_));
            }
            None => {
                // Record exports for the root module.
                debug!("(recording exports for module subtree) recording \
                        exports for root module `{}`",
                       self.module_to_str(module_));
            }
            Some(_) => {
                // Bail out.
                debug!("(recording exports for module subtree) not recording \
                        exports for `{}`",
                       self.module_to_str(module_));
                return;
            }
        }

        self.record_exports_for_module(module_);
        self.populate_module_if_necessary(module_);

        {
            let children = module_.children.borrow();
            for (_, &child_name_bindings) in children.get().iter() {
                match child_name_bindings.get_module_if_available() {
                    None => {
                        // Nothing to do.
                    }
                    Some(child_module) => {
                        self.record_exports_for_module_subtree(child_module);
                    }
                }
            }
        }

        let anonymous_children = module_.anonymous_children.borrow();
        for (_, &child_module) in anonymous_children.get().iter() {
            self.record_exports_for_module_subtree(child_module);
        }
    }

    fn record_exports_for_module(&mut self, module_: @Module) {
        let mut exports2 = ~[];

        self.add_exports_for_module(&mut exports2, module_);
        match module_.def_id.get() {
            Some(def_id) => {
                let mut export_map2 = self.export_map2.borrow_mut();
                export_map2.get().insert(def_id.node, exports2);
                debug!("(computing exports) writing exports for {} (some)",
                       def_id.node);
            }
            None => {}
        }
    }

    fn add_exports_of_namebindings(&mut self,
                                   exports2: &mut ~[Export2],
                                   name: Name,
                                   namebindings: @NameBindings,
                                   ns: Namespace,
                                   reexport: bool) {
        match namebindings.def_for_namespace(ns) {
            Some(d) => {
                debug!("(computing exports) YES: {} '{}' => {:?}",
                       if reexport { ~"reexport" } else { ~"export"},
                       interner_get(name),
                       def_id_of_def(d));
                exports2.push(Export2 {
                    reexport: reexport,
                    name: interner_get(name),
                    def_id: def_id_of_def(d)
                });
            }
            d_opt => {
                debug!("(computing reexports) NO: {:?}", d_opt);
            }
        }
    }

    fn add_exports_for_module(&mut self,
                              exports2: &mut ~[Export2],
                              module_: @Module) {
        let import_resolutions = module_.import_resolutions.borrow();
        for (name, importresolution) in import_resolutions.get().iter() {
            if !importresolution.is_public.get() {
                continue
            }
            let xs = [TypeNS, ValueNS];
            for &ns in xs.iter() {
                match importresolution.target_for_namespace(ns) {
                    Some(target) => {
                        debug!("(computing exports) maybe reexport '{}'",
                               interner_get(*name));
                        self.add_exports_of_namebindings(exports2,
                                                         *name,
                                                         target.bindings,
                                                         ns,
                                                         true)
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
        let orig_module = self.current_module;

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                self.populate_module_if_necessary(orig_module);

                let children = orig_module.children.borrow();
                match children.get().find(&name.name) {
                    None => {
                        debug!("!!! (with scope) didn't find `{}` in `{}`",
                               self.session.str_of(name),
                               self.module_to_str(orig_module));
                    }
                    Some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            None => {
                                debug!("!!! (with scope) didn't find module \
                                        for `{}` in `{}`",
                                       self.session.str_of(name),
                                       self.module_to_str(orig_module));
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
    fn upvarify(&mut self,
                    ribs: &mut ~[@Rib],
                    rib_index: uint,
                    def_like: DefLike,
                    span: Span,
                    allow_capturing_self: AllowCapturingSelfFlag)
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
            DlDef(d @ DefSelf(..))
                    if allow_capturing_self == DontAllowCapturingSelf => {
                def = d;
                is_ty_param = false;
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
                        let def_map = self.def_map.borrow();
                        def_map.get().find(&did.node).map(|x| *x)
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
                                              "attempt to use a type \
                                              argument out of scope");
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
                                              "attempt to use a type \
                                              argument out of scope");
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

    fn search_ribs(&mut self,
                       ribs: &mut ~[@Rib],
                       name: Name,
                       span: Span,
                       allow_capturing_self: AllowCapturingSelfFlag)
                       -> Option<DefLike> {
        // FIXME #4950: This should not use a while loop.
        // FIXME #4950: Try caching?

        let mut i = ribs.len();
        while i != 0 {
            i -= 1;
            let binding_opt = {
                let bindings = ribs[i].bindings.borrow();
                bindings.get().find_copy(&name)
            };
            match binding_opt {
                Some(def_like) => {
                    return self.upvarify(ribs, i, def_like, span,
                                         allow_capturing_self);
                }
                None => {
                    // Continue.
                }
            }
        }

        return None;
    }

    fn resolve_crate(&mut self, crate: &ast::Crate) {
        debug!("(resolving crate) starting");

        visit::walk_crate(self, crate, ());
    }

    fn resolve_item(&mut self, item: @item) {
        debug!("(resolving item) resolving {}",
               self.session.str_of(item.ident));

        match item.node {

            // enum item: resolve all the variants' discrs,
            // then resolve the ty params
            item_enum(ref enum_def, ref generics) => {
                for variant in (*enum_def).variants.iter() {
                    for dis_expr in variant.node.disr_expr.iter() {
                        // resolve the discriminator expr
                        // as a constant
                        self.with_constant_rib(|this| {
                            this.resolve_expr(*dis_expr);
                        });
                    }
                }

                // n.b. the discr expr gets visted twice.
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

            item_ty(_, ref generics) => {
                self.with_type_parameter_rib(HasTypeParameters(generics,
                                                               item.id,
                                                               0,
                                                               NormalRibKind),
                                             |this| {
                    visit::walk_item(this, item, ());
                });
            }

            item_impl(ref generics,
                      ref implemented_traits,
                      self_type,
                      ref methods) => {
                self.resolve_implementation(item.id,
                                            generics,
                                            implemented_traits,
                                            self_type,
                                            *methods);
            }

            item_trait(ref generics, ref traits, ref methods) => {
                // Create a new rib for the self type.
                let self_type_rib = @Rib::new(NormalRibKind);
                {
                    let mut type_ribs = self.type_ribs.borrow_mut();
                    type_ribs.get().push(self_type_rib);
                }
                // plain insert (no renaming)
                let name = self.type_self_ident.name;
                {
                    let mut bindings = self_type_rib.bindings.borrow_mut();
                    bindings.get().insert(name, DlDef(DefSelfTy(item.id)));
                }

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
                          required(ref ty_m) => {
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
                          provided(m) => {
                              this.resolve_method(MethodRibKind(item.id,
                                                     Provided(m.id)),
                                                  m,
                                                  generics.ty_params.len())
                          }
                        }
                    }
                });

                let mut type_ribs = self.type_ribs.borrow_mut();
                type_ribs.get().pop();
            }

            item_struct(ref struct_def, ref generics) => {
                self.resolve_struct(item.id,
                                    generics,
                                    struct_def.fields);
            }

            item_mod(ref module_) => {
                self.with_scope(Some(item.ident), |this| {
                    this.resolve_module(module_, item.span, item.ident,
                                        item.id);
                });
            }

            item_foreign_mod(ref foreign_module) => {
                self.with_scope(Some(item.ident), |this| {
                    for foreign_item in foreign_module.items.iter() {
                        match foreign_item.node {
                            foreign_item_fn(_, ref generics) => {
                                this.with_type_parameter_rib(
                                    HasTypeParameters(
                                        generics, foreign_item.id, 0,
                                        NormalRibKind),
                                    |this| visit::walk_foreign_item(this,
                                                                *foreign_item,
                                                                ()));
                            }
                            foreign_item_static(..) => {
                                visit::walk_foreign_item(this,
                                                         *foreign_item,
                                                         ());
                            }
                        }
                    }
                });
            }

            item_fn(fn_decl, _, _, ref generics, block) => {
                self.resolve_function(OpaqueFunctionRibKind,
                                      Some(fn_decl),
                                      HasTypeParameters
                                        (generics,
                                         item.id,
                                         0,
                                         OpaqueFunctionRibKind),
                                      block,
                                      NoSelfBinding);
            }

            item_static(..) => {
                self.with_constant_rib(|this| {
                    visit::walk_item(this, item, ());
                });
            }

          item_mac(..) => {
            fail!("item macros unimplemented")
          }
        }
    }

    fn with_type_parameter_rib(&mut self,
                               type_parameters: TypeParameters,
                               f: |&mut Resolver|) {
        match type_parameters {
            HasTypeParameters(generics, node_id, initial_index,
                              rib_kind) => {

                let function_type_rib = @Rib::new(rib_kind);
                {
                    let mut type_ribs = self.type_ribs.borrow_mut();
                    type_ribs.get().push(function_type_rib);
                }

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
                                    (DefTyParamBinder(node_id), AllPublic));
                    // plain insert (no renaming)
                    let mut bindings = function_type_rib.bindings
                                                        .borrow_mut();
                    bindings.get().insert(ident.name, def_like);
                }
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f(self);

        match type_parameters {
            HasTypeParameters(..) => {
                let mut type_ribs = self.type_ribs.borrow_mut();
                type_ribs.get().pop();
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }
    }

    fn with_label_rib(&mut self, f: |&mut Resolver|) {
        {
            let mut label_ribs = self.label_ribs.borrow_mut();
            label_ribs.get().push(@Rib::new(NormalRibKind));
        }

        f(self);

        {
            let mut label_ribs = self.label_ribs.borrow_mut();
            label_ribs.get().pop();
        }
    }

    fn with_constant_rib(&mut self, f: |&mut Resolver|) {
        {
            let mut value_ribs = self.value_ribs.borrow_mut();
            let mut type_ribs = self.type_ribs.borrow_mut();
            value_ribs.get().push(@Rib::new(ConstantItemRibKind));
            type_ribs.get().push(@Rib::new(ConstantItemRibKind));
        }
        f(self);
        {
            let mut value_ribs = self.value_ribs.borrow_mut();
            let mut type_ribs = self.type_ribs.borrow_mut();
            type_ribs.get().pop();
            value_ribs.get().pop();
        }
    }

    fn resolve_function(&mut self,
                            rib_kind: RibKind,
                            optional_declaration: Option<P<fn_decl>>,
                            type_parameters: TypeParameters,
                            block: P<Block>,
                            self_binding: SelfBinding) {
        // Create a value rib for the function.
        let function_value_rib = @Rib::new(rib_kind);
        {
            let mut value_ribs = self.value_ribs.borrow_mut();
            value_ribs.get().push(function_value_rib);
        }

        // Create a label rib for the function.
        {
            let mut label_ribs = self.label_ribs.borrow_mut();
            let function_label_rib = @Rib::new(rib_kind);
            label_ribs.get().push(function_label_rib);
        }

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

            // Add self to the rib, if necessary.
            match self_binding {
                NoSelfBinding => {
                    // Nothing to do.
                }
                HasSelfBinding(self_node_id, explicit_self) => {
                    let mutable = match explicit_self.node {
                        sty_uniq(m) | sty_value(m) if m == MutMutable => true,
                        _ => false
                    };
                    let def_like = DlDef(DefSelf(self_node_id, mutable));
                    function_value_rib.self_binding.set(Some(def_like));
                }
            }

            // Add each argument to the rib.
            match optional_declaration {
                None => {
                    // Nothing to do.
                }
                Some(declaration) => {
                    for argument in declaration.inputs.iter() {
                        let binding_mode = ArgumentIrrefutableMode;
                        this.resolve_pattern(argument.pat,
                                             binding_mode,
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

        let mut label_ribs = self.label_ribs.borrow_mut();
        label_ribs.get().pop();

        let mut value_ribs = self.value_ribs.borrow_mut();
        value_ribs.get().pop();
    }

    fn resolve_type_parameters(&mut self,
                                   type_parameters: &OptVec<TyParam>) {
        for type_parameter in type_parameters.iter() {
            for bound in type_parameter.bounds.iter() {
                self.resolve_type_parameter_bound(type_parameter.id, bound);
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
            RegionTyParamBound => {}
        }
    }

    fn resolve_trait_reference(&mut self,
                                   id: NodeId,
                                   trait_reference: &trait_ref,
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
                          fields: &[struct_field]) {
        let mut ident_map: HashMap<ast::Ident, &struct_field> = HashMap::new();
        for field in fields.iter() {
            match field.node.kind {
                named_field(ident, _) => {
                    match ident_map.find(&ident) {
                        Some(&prev_field) => {
                            let ident_str = self.session.str_of(ident);
                            self.resolve_error(field.span,
                                format!("field `{}` is already declared", ident_str));
                            self.session.span_note(prev_field.span,
                                "Previously declared here");
                        },
                        None => {
                            ident_map.insert(ident, field);
                        }
                    }
                }
                _ => ()
            }
        }

        // If applicable, create a rib for the type parameters.
        self.with_type_parameter_rib(HasTypeParameters(generics,
                                                       id,
                                                       0,
                                                       OpaqueFunctionRibKind),
                                     |this| {
            // Resolve the type parameters.
            this.resolve_type_parameters(&generics.ty_params);

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
                          method: @method,
                          outer_type_parameter_count: uint) {
        let method_generics = &method.generics;
        let type_parameters =
            HasTypeParameters(method_generics,
                              method.id,
                              outer_type_parameter_count,
                              rib_kind);
        // we only have self ty if it is a non static method
        let self_binding = match method.explicit_self.node {
          sty_static => { NoSelfBinding }
          _ => { HasSelfBinding(method.self_id, method.explicit_self) }
        };

        self.resolve_function(rib_kind,
                              Some(method.decl),
                              type_parameters,
                              method.body,
                              self_binding);
    }

    fn resolve_implementation(&mut self,
                                  id: NodeId,
                                  generics: &Generics,
                                  opt_trait_reference: &Option<trait_ref>,
                                  self_type: &Ty,
                                  methods: &[@method]) {
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
            let original_trait_refs;
            match opt_trait_reference {
                &Some(ref trait_reference) => {
                    this.resolve_trait_reference(id, trait_reference,
                        TraitImplementation);

                    // Record the current set of trait references.
                    let mut new_trait_refs = ~[];
                    {
                        let def_map = this.def_map.borrow();
                        let r = def_map.get().find(&trait_reference.ref_id);
                        for &def in r.iter() {
                            new_trait_refs.push(def_id_of_def(*def));
                        }
                    }
                    original_trait_refs = Some(util::replace(
                        &mut this.current_trait_refs,
                        Some(new_trait_refs)));
                }
                &None => {
                    original_trait_refs = None;
                }
            }

            // Resolve the self type.
            this.resolve_type(self_type);

            for method in methods.iter() {
                // We also need a new scope for the method-specific
                // type parameters.
                this.resolve_method(MethodRibKind(
                    id,
                    Provided(method.id)),
                    *method,
                    outer_type_parameter_count);
/*
                    let borrowed_type_parameters = &method.tps;
                    self.resolve_function(MethodRibKind(
                                          id,
                                          Provided(method.id)),
                                          Some(method.decl),
                                          HasTypeParameters
                                            (borrowed_type_parameters,
                                             method.id,
                                             outer_type_parameter_count,
                                             NormalRibKind),
                                          method.body,
                                          HasSelfBinding(method.self_id),
                                          visitor);
*/
            }

            // Restore the original trait references.
            match original_trait_refs {
                Some(r) => { this.current_trait_refs = r; }
                None => ()
            }
        });
    }

    fn resolve_module(&mut self,
                          module_: &_mod,
                          _span: Span,
                          _name: Ident,
                          id: NodeId) {
        // Write the implementations in scope into the module metadata.
        debug!("(resolving module) resolving module ID {}", id);
        visit::walk_mod(self, module_, ());
    }

    fn resolve_local(&mut self, local: @Local) {
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
    fn binding_mode_map(&mut self, pat: @Pat) -> BindingMap {
        let mut result = HashMap::new();
        pat_bindings(self.def_map, pat, |binding_mode, _id, sp, path| {
            let name = mtwt_resolve(path_to_ident(path));
            result.insert(name,
                          binding_info {span: sp,
                                        binding_mode: binding_mode});
        });
        return result;
    }

    // check that all of the arms in an or-pattern have exactly the
    // same set of bindings, with the same binding modes for each.
    fn check_consistent_bindings(&mut self, arm: &Arm) {
        if arm.pats.len() == 0 { return; }
        let map_0 = self.binding_mode_map(arm.pats[0]);
        for (i, p) in arm.pats.iter().enumerate() {
            let map_i = self.binding_mode_map(*p);

            for (&key, &binding_0) in map_0.iter() {
                match map_i.find(&key) {
                  None => {
                    self.resolve_error(
                        p.span,
                        format!("variable `{}` from pattern \\#1 is \
                                  not bound in pattern \\#{}",
                             interner_get(key), i + 1));
                  }
                  Some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.resolve_error(
                            binding_i.span,
                            format!("variable `{}` is bound with different \
                                      mode in pattern \\#{} than in pattern \\#1",
                                 interner_get(key), i + 1));
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
                             interner_get(key), i + 1));
                }
            }
        }
    }

    fn resolve_arm(&mut self, arm: &Arm) {
        {
            let mut value_ribs = self.value_ribs.borrow_mut();
            value_ribs.get().push(@Rib::new(NormalRibKind));
        }

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
        self.resolve_block(arm.body);

        let mut value_ribs = self.value_ribs.borrow_mut();
        value_ribs.get().pop();
    }

    fn resolve_block(&mut self, block: P<Block>) {
        debug!("(resolving block) entering block");
        {
            let mut value_ribs = self.value_ribs.borrow_mut();
            value_ribs.get().push(@Rib::new(NormalRibKind));
        }

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        let anonymous_children = self.current_module
                                     .anonymous_children
                                     .borrow();
        match anonymous_children.get().find(&block.id) {
            None => { /* Nothing to do. */ }
            Some(&anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module;
            }
        }

        // Descend into the block.
        visit::walk_block(self, block, ());

        // Move back up.
        self.current_module = orig_module;

        let mut value_ribs = self.value_ribs.borrow_mut();
        value_ribs.get().pop();
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(&mut self, ty: &Ty) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            ty_path(ref path, ref bounds, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // scopes looking for it.
                let mut result_def = None;

                // First, check to see whether the name is a primitive type.
                if path.segments.len() == 1 {
                    let id = path.segments.last().identifier;

                    match self.primitive_type_table
                            .primitive_types
                            .find(&id.name) {

                        Some(&primitive_type) => {
                            result_def =
                                Some((DefPrimTy(primitive_type), AllPublic));

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
                                       self.session.str_of(path.segments
                                                               .last()
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

            ty_closure(c) => {
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
                       pattern: @Pat,
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

                    let ident = path.segments[0].identifier;
                    let renamed = mtwt_resolve(ident);

                    match self.resolve_bare_identifier_pattern(ident) {
                        FoundStructOrEnumVariant(def, lp)
                                if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    struct or enum variant",
                                   interner_get(renamed));

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
                                                       interner_get(renamed)));
                        }
                        FoundConst(def, lp) if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `{}` to \
                                    constant",
                                   interner_get(renamed));

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
                                   interner_get(renamed));

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

                            self.record_def(pattern.id, (def, AllPublic));

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)

                            match bindings_list {
                                Some(ref mut bindings_list)
                                if !bindings_list.contains_key(&renamed) => {
                                    let this = &mut *self;
                                    {
                                        let mut value_ribs =
                                            this.value_ribs.borrow_mut();
                                        let last_rib = value_ribs.get()[
                                            value_ribs.get().len() - 1];
                                        let mut bindings =
                                            last_rib.bindings.borrow_mut();
                                        bindings.get().insert(renamed,
                                                              DlDef(def));
                                    }
                                    bindings_list.insert(renamed, pat_id);
                                }
                                Some(ref mut b) => {
                                  if b.find(&renamed) == Some(&pat_id) {
                                      // Then this is a duplicate variable
                                      // in the same disjunct, which is an
                                      // error
                                     self.resolve_error(pattern.span,
                                       format!("Identifier `{}` is bound more \
                                             than once in the same pattern",
                                            path_to_str(path, self.session
                                                        .intr())));
                                  }
                                  // Not bound in the same pattern: do nothing
                                }
                                None => {
                                    let this = &mut *self;
                                    {
                                        let mut value_ribs =
                                            this.value_ribs.borrow_mut();
                                        let last_rib = value_ribs.get()[
                                                value_ribs.get().len() - 1];
                                        let mut bindings =
                                            last_rib.bindings.borrow_mut();
                                        bindings.get().insert(renamed,
                                                              DlDef(def));
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
                                     self.session.str_of(
                                         path.segments.last().identifier)))
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
                            self.resolve_error(
                                path.span,
                                format!("`{}` is not an enum variant, struct or const",
                                     self.session
                                         .str_of(path.segments
                                                     .last()
                                                     .identifier)));
                        }
                        None => {
                            self.resolve_error(path.span,
                                               format!("unresolved enum variant, \
                                                    struct or const `{}`",
                                                    self.session
                                                        .str_of(path.segments
                                                                    .last()
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
                                if self.structs.contains(&class_id) => {
                            let class_def = DefStruct(class_id);
                            self.record_def(pattern.id, (class_def, lp));
                        }
                        Some(definition @ (DefStruct(class_id), _)) => {
                            assert!(self.structs.contains(&class_id));
                            self.record_def(pattern.id, definition);
                        }
                        Some(definition @ (DefVariant(_, variant_id, _), _))
                                if self.structs.contains(&variant_id) => {
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
                                           ->
                                           BareIdentifierPatternResolution {
        match self.resolve_item_in_lexical_scope(self.current_module,
                                                 name,
                                                 ValueNS,
                                                 SearchThroughModules) {
            Success((target, _)) => {
                debug!("(resolve bare identifier pattern) succeeded in \
                         finding {} at {:?}",
                        self.session.str_of(name),
                        target.bindings.value_def.get());
                match target.bindings.value_def.get() {
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
                                return FoundStructOrEnumVariant(def, AllPublic);
                            }
                            def @ DefStatic(_, false) => {
                                return FoundConst(def, AllPublic);
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
                        self.session.str_of(name));
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
                                            .last()
                                            .identifier,
                                        namespace,
                                        check_ribs,
                                        path.span);

        if path.segments.len() > 1 {
            let def = self.resolve_module_relative_path(path, namespace);
            match (def, unqualified_def) {
                (Some((d, _)), Some((ud, _))) if d == ud => {
                    self.session.add_lint(unnecessary_qualification,
                                          id,
                                          path.span,
                                          ~"unnecessary qualification");
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
                    return Some((def, AllPublic));
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
                                            containing_module: @Module,
                                            name: Ident,
                                            namespace: Namespace)
                                                -> NameDefinition {
        // First, search children.
        self.populate_module_if_necessary(containing_module);

        {
            let children = containing_module.children.borrow();
            match children.get().find(&name.name) {
                Some(child_name_bindings) => {
                    match child_name_bindings.def_for_namespace(namespace) {
                        Some(def) => {
                            // Found it. Stop the search here.
                            let p = child_name_bindings.defined_in_public_namespace(
                                            namespace);
                            let lp = if p {AllPublic} else {
                                DependsOn(def_id_of_def(def))
                            };
                            return ChildNameDefinition(def, lp);
                        }
                        None => {}
                    }
                }
                None => {}
            }
        }

        // Next, search import resolutions.
        let import_resolutions = containing_module.import_resolutions
                                                  .borrow();
        match import_resolutions.get().find(&name.name) {
            Some(import_resolution) if import_resolution.is_public.get() => {
                match (*import_resolution).target_for_namespace(namespace) {
                    Some(target) => {
                        match target.bindings.def_for_namespace(namespace) {
                            Some(def) => {
                                // Found it.
                                let id = import_resolution.id(namespace);
                                self.used_imports.insert(id);
                                return ImportNameDefinition(def, AllPublic);
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
            let module_opt = {
                let external_module_children =
                    containing_module.external_module_children.borrow();
                external_module_children.get().find_copy(&name.name)
            };
            match module_opt {
                None => {}
                Some(module) => {
                    match module.def_id.get() {
                        None => {} // Continue.
                        Some(def_id) => {
                            let lp = if module.is_public {AllPublic} else {
                                DependsOn(def_id)
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
        let module_path_idents = path.segments.init().map(|ps| ps.identifier);

        let containing_module;
        let last_private;
        match self.resolve_module_path(self.current_module,
                                       module_path_idents,
                                       UseLexicalScope,
                                       path.span,
                                       PathSearch) {
            Failed => {
                let msg = format!("use of undeclared module `{}`",
                                  self.idents_to_str(module_path_idents));
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

        let ident = path.segments.last().identifier;
        let def = match self.resolve_definition_of_name_in_module(containing_module,
                                                        ident,
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
                let method_map = self.method_map.borrow();
                match method_map.get().find(&ident.name) {
                    Some(s) => {
                        match containing_module.def_id.get() {
                            Some(def_id) if s.contains(&def_id) => {
                                debug!("containing module was a trait or impl \
                                        and name was a method -> not resolved");
                                return None;
                            },
                            _ => (),
                        }
                    },
                    None => (),
                }
            },
            _ => (),
        };
        return Some(def);
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path(&mut self,
                                   path: &Path,
                                   namespace: Namespace)
                                       -> Option<(Def, LastPrivate)> {
        let module_path_idents = path.segments.init().map(|ps| ps.identifier);

        let root_module = self.graph_root.get_module();

        let containing_module;
        let last_private;
        match self.resolve_module_path_from_root(root_module,
                                                 module_path_idents,
                                                 0,
                                                 path.span,
                                                 PathSearch,
                                                 AllPublic) {
            Failed => {
                let msg = format!("use of undeclared module `::{}`",
                                  self.idents_to_str(module_path_idents));
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

        let name = path.segments.last().identifier;
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
        let search_result;
        match namespace {
            ValueNS => {
                let renamed = mtwt_resolve(ident);
                let mut value_ribs = self.value_ribs.borrow_mut();
                search_result = self.search_ribs(value_ribs.get(),
                                                 renamed,
                                                 span,
                                                 DontAllowCapturingSelf);
            }
            TypeNS => {
                let name = ident.name;
                let mut type_ribs = self.type_ribs.borrow_mut();
                search_result = self.search_ribs(type_ribs.get(),
                                                 name,
                                                 span,
                                                 AllowCapturingSelf);
            }
        }

        match search_result {
            Some(DlDef(def)) => {
                debug!("(resolving path in local ribs) resolved `{}` to \
                        local: {:?}",
                       self.session.str_of(ident),
                       def);
                return Some(def);
            }
            Some(DlField) | Some(DlImpl(_)) | None => {
                return None;
            }
        }
    }

    fn resolve_self_value_in_local_ribs(&mut self, span: Span)
                                            -> Option<Def> {
        // FIXME #4950: This should not use a while loop.
        let mut i = {
            let value_ribs = self.value_ribs.borrow();
            value_ribs.get().len()
        };
        while i != 0 {
            i -= 1;
            let self_binding_opt = {
                let value_ribs = self.value_ribs.borrow();
                value_ribs.get()[i].self_binding.get()
            };
            match self_binding_opt {
                Some(def_like) => {
                    let mut value_ribs = self.value_ribs.borrow_mut();
                    match self.upvarify(value_ribs.get(),
                                        i,
                                        def_like,
                                        span,
                                        DontAllowCapturingSelf) {
                        Some(DlDef(def)) => return Some(def),
                        _ => {
                            if self.session.has_errors() {
                                // May happen inside a nested fn item, cf #6642.
                                return None;
                            } else {
                                self.session.span_bug(span,
                                        "self wasn't mapped to a def?!")
                            }
                        }
                    }
                }
                None => {}
            }
        }

        None
    }

    fn resolve_item_by_identifier_in_lexical_scope(&mut self,
                                                   ident: Ident,
                                                   namespace: Namespace)
                                                -> Option<(Def, LastPrivate)> {
        // Check the items.
        match self.resolve_item_in_lexical_scope(self.current_module,
                                                 ident,
                                                 namespace,
                                                 DontSearchThroughModules) {
            Success((target, _)) => {
                match (*target.bindings).def_for_namespace(namespace) {
                    None => {
                        // This can happen if we were looking for a type and
                        // found a module instead. Modules don't have defs.
                        debug!("(resolving item path by identifier in lexical \
                                 scope) failed to resolve {} after success...",
                                 self.session.str_of(ident));
                        return None;
                    }
                    Some(def) => {
                        debug!("(resolving item path in lexical scope) \
                                resolved `{}` to item",
                               self.session.str_of(ident));
                        // This lookup is "all public" because it only searched
                        // for one identifier in the current module (couldn't
                        // have passed through reexports or anything like that.
                        return Some((def, AllPublic));
                    }
                }
            }
            Indeterminate => {
                fail!("unexpected indeterminate result");
            }
            Failed => {
                debug!("(resolving item path by identifier in lexical scope) \
                         failed to resolve {}", self.session.str_of(ident));
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

    fn resolve_error(&mut self, span: Span, s: &str) {
        if self.emit_errors {
            self.session.span_err(span, s);
        }
    }

    fn find_best_match_for_name(&mut self, name: &str, max_distance: uint)
                                -> Option<@str> {
        let this = &mut *self;

        let mut maybes: ~[@str] = ~[];
        let mut values: ~[uint] = ~[];

        let mut j = {
            let value_ribs = this.value_ribs.borrow();
            value_ribs.get().len()
        };
        while j != 0 {
            j -= 1;
            let value_ribs = this.value_ribs.borrow();
            let bindings = value_ribs.get()[j].bindings.borrow();
            for (&k, _) in bindings.get().iter() {
                maybes.push(interner_get(k));
                values.push(uint::max_value);
            }
        }

        let mut smallest = 0;
        for (i, &other) in maybes.iter().enumerate() {
            values[i] = name.lev_distance(other);

            if values[i] <= values[smallest] {
                smallest = i;
            }
        }

        if values.len() > 0 &&
            values[smallest] != uint::max_value &&
            values[smallest] < name.len() + 2 &&
            values[smallest] <= max_distance &&
            name != maybes[smallest] {

            Some(maybes.swap_remove(smallest))

        } else {
            None
        }
    }

    fn resolve_expr(&mut self, expr: @Expr) {
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
                              if self.structs.contains(&struct_id) => {
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
                            _ =>
                               // limit search to 5 to reduce the number
                               // of stupid suggestions
                               match self.find_best_match_for_name(wrong_name, 5) {
                                   Some(m) => {
                                       self.resolve_error(expr.span,
                                           format!("unresolved name `{}`. \
                                                    Did you mean `{}`?",
                                                    wrong_name, m));
                                   }
                                   None => {
                                       self.resolve_error(expr.span,
                                            format!("unresolved name `{}`.",
                                                    wrong_name));
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
                                      Some(fn_decl),
                                      NoTypeParameters,
                                      block,
                                      NoSelfBinding);
            }

            ExprStruct(ref path, _, _) => {
                // Resolve the path to the structure it goes to.
                match self.resolve_path(expr.id, path, TypeNS, false) {
                    Some((DefTy(class_id), lp)) | Some((DefStruct(class_id), lp))
                            if self.structs.contains(&class_id) => {
                        let class_def = DefStruct(class_id);
                        self.record_def(expr.id, (class_def, lp));
                    }
                    Some(definition @ (DefVariant(_, class_id, _), _))
                            if self.structs.contains(&class_id) => {
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
                    // plain insert (no renaming)
                    {
                        let mut label_ribs = this.label_ribs.borrow_mut();
                        let rib = label_ribs.get()[label_ribs.get().len() -
                                                   1];
                        let mut bindings = rib.bindings.borrow_mut();
                        bindings.get().insert(label.name, def_like);
                    }

                    visit::walk_expr(this, expr, ());
                })
            }

            ExprForLoop(..) => fail!("non-desugared expr_for_loop"),

            ExprBreak(Some(label)) | ExprAgain(Some(label)) => {
                let mut label_ribs = self.label_ribs.borrow_mut();
                match self.search_ribs(label_ribs.get(), label, expr.span,
                                       DontAllowCapturingSelf) {
                    None =>
                        self.resolve_error(expr.span,
                                              format!("use of undeclared label \
                                                   `{}`",
                                                   interner_get(label))),
                    Some(DlDef(def @ DefLabel(_))) => {
                        // XXX: is AllPublic correct?
                        self.record_def(expr.id, (def, AllPublic))
                    }
                    Some(_) => {
                        self.session.span_bug(expr.span,
                                              "label wasn't mapped to a \
                                               label def!")
                    }
                }
            }

            ExprSelf => {
                match self.resolve_self_value_in_local_ribs(expr.span) {
                    None => {
                        self.resolve_error(expr.span,
                                              "`self` is not allowed in \
                                               this context")
                    }
                    Some(def) => self.record_def(expr.id, (def, AllPublic)),
                }
            }

            _ => {
                visit::walk_expr(self, expr, ());
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(&mut self,
                                                         expr: @Expr) {
        match expr.node {
            ExprField(_, ident, _) => {
                // FIXME(#6890): Even though you can't treat a method like a
                // field, we need to add any trait methods we find that match
                // the field name so that we can do some nice error reporting
                // later on in typeck.
                let traits = self.search_for_traits_containing_method(ident);
                self.trait_map.insert(expr.id, @RefCell::new(traits));
            }
            ExprMethodCall(_, _, ident, _, _, _) => {
                debug!("(recording candidate traits for expr) recording \
                        traits for {}",
                       expr.id);
                let traits = self.search_for_traits_containing_method(ident);
                self.trait_map.insert(expr.id, @RefCell::new(traits));
            }
            ExprBinary(_, BiAdd, _, _) | ExprAssignOp(_, BiAdd, _, _) => {
                let i = self.lang_items.add_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiSub, _, _) | ExprAssignOp(_, BiSub, _, _) => {
                let i = self.lang_items.sub_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiMul, _, _) | ExprAssignOp(_, BiMul, _, _) => {
                let i = self.lang_items.mul_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiDiv, _, _) | ExprAssignOp(_, BiDiv, _, _) => {
                let i = self.lang_items.div_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiRem, _, _) | ExprAssignOp(_, BiRem, _, _) => {
                let i = self.lang_items.rem_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiBitXor, _, _) | ExprAssignOp(_, BiBitXor, _, _) => {
                let i = self.lang_items.bitxor_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiBitAnd, _, _) | ExprAssignOp(_, BiBitAnd, _, _) => {
                let i = self.lang_items.bitand_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiBitOr, _, _) | ExprAssignOp(_, BiBitOr, _, _) => {
                let i = self.lang_items.bitor_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiShl, _, _) | ExprAssignOp(_, BiShl, _, _) => {
                let i = self.lang_items.shl_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiShr, _, _) | ExprAssignOp(_, BiShr, _, _) => {
                let i = self.lang_items.shr_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiLt, _, _) | ExprBinary(_, BiLe, _, _) |
            ExprBinary(_, BiGe, _, _) | ExprBinary(_, BiGt, _, _) => {
                let i = self.lang_items.ord_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprBinary(_, BiEq, _, _) | ExprBinary(_, BiNe, _, _) => {
                let i = self.lang_items.eq_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprUnary(_, UnNeg, _) => {
                let i = self.lang_items.neg_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprUnary(_, UnNot, _) => {
                let i = self.lang_items.not_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            ExprIndex(..) => {
                let i = self.lang_items.index_trait();
                self.add_fixed_trait_for_expr(expr.id, i);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn search_for_traits_containing_method(&mut self, name: Ident)
                                               -> ~[DefId] {
        debug!("(searching for traits containing method) looking for '{}'",
               self.session.str_of(name));

        let mut found_traits = ~[];
        let mut search_module = self.current_module;
        let method_map = self.method_map.borrow();
        match method_map.get().find(&name.name) {
            Some(candidate_traits) => loop {
                // Look for the current trait.
                match self.current_trait_refs {
                    Some(ref trait_def_ids) => {
                        for trait_def_id in trait_def_ids.iter() {
                            if candidate_traits.contains(trait_def_id) {
                                self.add_trait_info(&mut found_traits,
                                                    *trait_def_id,
                                                    name);
                            }
                        }
                    }
                    None => {
                        // Nothing to do.
                    }
                }

                // Look for trait children.
                self.populate_module_if_necessary(search_module);

                let children = search_module.children.borrow();
                for (_, &child_name_bindings) in children.get().iter() {
                    match child_name_bindings.def_for_namespace(TypeNS) {
                        Some(def) => {
                            match def {
                                DefTrait(trait_def_id) => {
                                    if candidate_traits.contains(&trait_def_id) {
                                        self.add_trait_info(
                                            &mut found_traits,
                                            trait_def_id, name);
                                    }
                                }
                                _ => {
                                    // Continue.
                                }
                            }
                        }
                        None => {
                            // Continue.
                        }
                    }
                }

                // Look for imports.
                let import_resolutions = search_module.import_resolutions
                                                      .borrow();
                for (_, &import_resolution) in import_resolutions.get()
                                                                 .iter() {
                    match import_resolution.target_for_namespace(TypeNS) {
                        None => {
                            // Continue.
                        }
                        Some(target) => {
                            match target.bindings.def_for_namespace(TypeNS) {
                                Some(def) => {
                                    match def {
                                        DefTrait(trait_def_id) => {
                                            if candidate_traits.contains(&trait_def_id) {
                                                self.add_trait_info(
                                                    &mut found_traits,
                                                    trait_def_id, name);
                                                self.used_imports.insert(
                                                    import_resolution.type_id
                                                                     .get());
                                            }
                                        }
                                        _ => {
                                            // Continue.
                                        }
                                    }
                                }
                                None => {
                                    // Continue.
                                }
                            }
                        }
                    }
                }

                // Move to the next parent.
                match search_module.parent_link {
                    NoParentLink => {
                        // Done.
                        break;
                    }
                    ModuleParentLink(parent_module, _) |
                    BlockParentLink(parent_module, _) => {
                        search_module = parent_module;
                    }
                }
            },
            _ => ()
        }

        return found_traits;
    }

    fn add_trait_info(&self,
                          found_traits: &mut ~[DefId],
                          trait_def_id: DefId,
                          name: Ident) {
        debug!("(adding trait info) found trait {}:{} for method '{}'",
               trait_def_id.crate,
               trait_def_id.node,
               self.session.str_of(name));
        found_traits.push(trait_def_id);
    }

    fn add_fixed_trait_for_expr(&mut self,
                                    expr_id: NodeId,
                                    trait_id: Option<DefId>) {
        match trait_id {
            Some(trait_id) => {
                self.trait_map.insert(expr_id, @RefCell::new(~[trait_id]));
            }
            None => {}
        }
    }

    fn record_def(&mut self, node_id: NodeId, (def, lp): (Def, LastPrivate)) {
        debug!("(recording def) recording {:?} for {:?}, last private {:?}",
                def, node_id, lp);
        self.last_private.insert(node_id, lp);
        let mut def_map = self.def_map.borrow_mut();
        def_map.get().insert_or_update_with(node_id, def, |_, old_value| {
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
    // Although this is a lint pass, it lives in here because it depends on
    // resolve data structures.
    //

    fn check_for_unused_imports(&self, crate: &ast::Crate) {
        let mut visitor = UnusedImportCheckVisitor{ resolver: self };
        visit::walk_crate(&mut visitor, crate, ());
    }

    fn check_for_item_unused_imports(&self, vi: &view_item) {
        // Ignore is_public import statements because there's no way to be sure
        // whether they're used or not. Also ignore imports with a dummy span
        // because this means that they were generated in some fashion by the
        // compiler and we don't need to consider them.
        if vi.vis == public { return }
        if vi.span == DUMMY_SP { return }

        match vi.node {
            view_item_extern_mod(..) => {} // ignore
            view_item_use(ref path) => {
                for p in path.iter() {
                    match p.node {
                        view_path_simple(_, _, id) | view_path_glob(_, id) => {
                            if !self.used_imports.contains(&id) {
                                self.session.add_lint(unused_imports,
                                                      id, p.span,
                                                      ~"unused import");
                            }
                        }

                        view_path_list(_, ref list, _) => {
                            for i in list.iter() {
                                if !self.used_imports.contains(&i.node.id) {
                                    self.session.add_lint(unused_imports,
                                                          i.node.id, i.span,
                                                          ~"unused import");
                                }
                            }
                        }
                    }
                }
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
    fn module_to_str(&mut self, module_: @Module) -> ~str {
        let mut idents = ~[];
        let mut current_module = module_;
        loop {
            match current_module.parent_link {
                NoParentLink => {
                    break;
                }
                ModuleParentLink(module_, name) => {
                    idents.push(name);
                    current_module = module_;
                }
                BlockParentLink(module_, _) => {
                    idents.push(special_idents::opaque);
                    current_module = module_;
                }
            }
        }

        if idents.len() == 0 {
            return ~"???";
        }
        return self.idents_to_str(idents.move_rev_iter().collect::<~[ast::Ident]>());
    }

    #[allow(dead_code)]   // useful for debugging
    fn dump_module(&mut self, module_: @Module) {
        debug!("Dump of module `{}`:", self.module_to_str(module_));

        debug!("Children:");
        self.populate_module_if_necessary(module_);
        let children = module_.children.borrow();
        for (&name, _) in children.get().iter() {
            debug!("* {}", interner_get(name));
        }

        debug!("Import resolutions:");
        let import_resolutions = module_.import_resolutions.borrow();
        for (name, import_resolution) in import_resolutions.get().iter() {
            let value_repr;
            match import_resolution.target_for_namespace(ValueNS) {
                None => { value_repr = ~""; }
                Some(_) => {
                    value_repr = ~" value:?";
                    // FIXME #4954
                }
            }

            let type_repr;
            match import_resolution.target_for_namespace(TypeNS) {
                None => { type_repr = ~""; }
                Some(_) => {
                    type_repr = ~" type:?";
                    // FIXME #4954
                }
            }

            debug!("* {}:{}{}", interner_get(*name),
                   value_repr, type_repr);
        }
    }
}

pub struct CrateMap {
    def_map: DefMap,
    exp_map2: ExportMap2,
    trait_map: TraitMap,
    external_exports: ExternalExports,
    last_private_map: LastPrivateMap,
}

/// Entry point to crate resolution.
pub fn resolve_crate(session: Session,
                     lang_items: LanguageItems,
                     crate: &Crate)
                  -> CrateMap {
    let mut resolver = Resolver(session, lang_items, crate.span);
    resolver.resolve(crate);
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
