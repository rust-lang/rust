// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session;
use driver::session::Session;
use metadata::csearch::{each_path, get_trait_method_def_ids};
use metadata::csearch::get_method_name_and_self_ty;
use metadata::csearch::get_static_methods_if_impl;
use metadata::csearch::get_type_name_if_impl;
use metadata::cstore::find_extern_mod_stmt_cnum;
use metadata::decoder::{def_like, dl_def, dl_field, dl_impl};
use middle::lang_items::LanguageItems;
use middle::lint::{allow, level, unused_imports};
use middle::lint::{get_lint_level, get_lint_settings_level};
use middle::pat_util::pat_bindings;

use syntax::ast::{RegionTyParamBound, TraitTyParamBound, _mod, add, arm};
use syntax::ast::{binding_mode, bitand, bitor, bitxor, blk};
use syntax::ast::{bind_infer, bind_by_ref, bind_by_copy};
use syntax::ast::{crate, decl_item, def, def_arg, def_binding};
use syntax::ast::{def_const, def_foreign_mod, def_fn, def_id, def_label};
use syntax::ast::{def_local, def_mod, def_prim_ty, def_region, def_self};
use syntax::ast::{def_self_ty, def_static_method, def_struct, def_ty};
use syntax::ast::{def_ty_param, def_typaram_binder, def_trait};
use syntax::ast::{def_upvar, def_use, def_variant, expr, expr_assign_op};
use syntax::ast::{expr_binary, expr_break, expr_field};
use syntax::ast::{expr_fn_block, expr_index, expr_method_call, expr_path};
use syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
use syntax::ast::{def_upvar, def_use, def_variant, div, eq};
use syntax::ast::{expr, expr_again, expr_assign_op};
use syntax::ast::{expr_index, expr_loop};
use syntax::ast::{expr_path, expr_struct, expr_unary, fn_decl};
use syntax::ast::{foreign_item, foreign_item_const, foreign_item_fn, ge};
use syntax::ast::Generics;
use syntax::ast::{gt, ident, inherited, item, item_struct};
use syntax::ast::{item_const, item_enum, item_fn, item_foreign_mod};
use syntax::ast::{item_impl, item_mac, item_mod, item_trait, item_ty, le};
use syntax::ast::{local, local_crate, lt, method, mul};
use syntax::ast::{named_field, ne, neg, node_id, pat, pat_enum, pat_ident};
use syntax::ast::{Path, pat_lit, pat_range, pat_struct};
use syntax::ast::{prim_ty, private, provided};
use syntax::ast::{public, required, rem, self_ty_, shl, shr, stmt_decl};
use syntax::ast::{struct_dtor, struct_field, struct_variant_kind};
use syntax::ast::{sty_static, subtract, trait_ref, tuple_variant_kind, Ty};
use syntax::ast::{ty_bool, ty_char, ty_f, ty_f32, ty_f64, ty_float, ty_i};
use syntax::ast::{ty_i16, ty_i32, ty_i64, ty_i8, ty_int, TyParam, ty_path};
use syntax::ast::{ty_str, ty_u, ty_u16, ty_u32, ty_u64, ty_u8, ty_uint};
use syntax::ast::unnamed_field;
use syntax::ast::{variant, view_item, view_item_extern_mod};
use syntax::ast::{view_item_use, view_path_glob, view_path_list};
use syntax::ast::{view_path_simple, anonymous, named, not};
use syntax::ast::{unsafe_fn};
use syntax::ast_util::{def_id_of_def, local_def};
use syntax::ast_util::{path_to_ident, walk_pat, trait_method_to_ty_method};
use syntax::ast_util::{Privacy, Public, Private};
use syntax::ast_util::{variant_visibility_to_privacy, visibility_to_privacy};
use syntax::attr::{attr_metas, contains_name, attrs_contains_name};
use syntax::parse::token::ident_interner;
use syntax::parse::token::special_idents;
use syntax::print::pprust::path_to_str;
use syntax::codemap::{span, dummy_sp};
use syntax::visit::{default_visitor, mk_vt, Visitor, visit_block};
use syntax::visit::{visit_crate, visit_expr, visit_expr_opt};
use syntax::visit::{visit_foreign_item, visit_item};
use syntax::visit::{visit_mod, visit_ty, vt};
use syntax::opt_vec::OptVec;

use core::option::Some;
use core::str::each_split_str;
use core::hashmap::{HashMap, HashSet};
use core::util;

// Definition mapping
pub type DefMap = @mut HashMap<node_id,def>;

pub struct binding_info {
    span: span,
    binding_mode: binding_mode,
}

// Map from the name in a pattern to its binding mode.
pub type BindingMap = HashMap<ident,binding_info>;

// Implementation resolution
//
// FIXME #4946: This kind of duplicates information kept in
// ty::method. Maybe it should go away.

pub struct MethodInfo {
    did: def_id,
    n_tps: uint,
    ident: ident,
    self_type: self_ty_
}

pub struct Impl {
    did: def_id,
    ident: ident,
    methods: ~[@MethodInfo]
}

// Trait method resolution
pub type TraitMap = HashMap<node_id,@mut ~[def_id]>;

// This is the replacement export map. It maps a module to all of the exports
// within.
pub type ExportMap2 = @mut HashMap<node_id, ~[Export2]>;

pub struct Export2 {
    name: @~str,        // The name of the target.
    def_id: def_id,     // The definition of the target.
    reexport: bool,     // Whether this is a reexport.
}

#[deriving(Eq)]
pub enum PatternBindingMode {
    RefutableMode,
    LocalIrrefutableMode,
    ArgumentIrrefutableMode,
}

#[deriving(Eq)]
pub enum Namespace {
    TypeNS,
    ValueNS
}

/// A NamespaceResult represents the result of resolving an import in
/// a particular namespace. The result is either definitely-resolved,
/// definitely- unresolved, or unknown.
pub enum NamespaceResult {
    /// Means that resolve hasn't gathered enough information yet to determine
    /// whether the name is bound in this namespace. (That is, it hasn't
    /// resolved all `use` directives yet.)
    UnknownResult,
    /// Means that resolve has determined that the name is definitely
    /// not bound in the namespace.
    UnboundResult,
    /// Means that resolve has determined that the name is bound in the Module
    /// argument, and specified by the NameBindings argument.
    BoundResult(@mut Module, @mut NameBindings)
}

pub impl NamespaceResult {
    fn is_unknown(&self) -> bool {
        match *self {
            UnknownResult => true,
            _ => false
        }
    }
}

pub enum NameDefinition {
    NoNameDefinition,           //< The name was unbound.
    ChildNameDefinition(def),   //< The name identifies an immediate child.
    ImportNameDefinition(def)   //< The name identifies an import.
}

#[deriving(Eq)]
pub enum Mutability {
    Mutable,
    Immutable
}

pub enum SelfBinding {
    NoSelfBinding,
    HasSelfBinding(node_id, bool /* is implicit */)
}

pub type ResolveVisitor = vt<()>;

/// Contains data for specific types of import directives.
pub enum ImportDirectiveSubclass {
    SingleImport(ident /* target */, ident /* source */),
    GlobImport
}

/// The context that we thread through while building the reduced graph.
pub enum ReducedGraphParent {
    ModuleReducedGraphParent(@mut Module)
}

pub enum ResolveResult<T> {
    Failed,         // Failed to resolve the name.
    Indeterminate,  // Couldn't determine due to unresolved globs.
    Success(T)      // Successfully resolved the import.
}

pub impl<T> ResolveResult<T> {
    fn failed(&self) -> bool {
        match *self { Failed => true, _ => false }
    }
    fn indeterminate(&self) -> bool {
        match *self { Indeterminate => true, _ => false }
    }
}

pub enum TypeParameters<'self> {
    NoTypeParameters,                   //< No type parameters.
    HasTypeParameters(&'self Generics,  //< Type parameters.
                      node_id,          //< ID of the enclosing item

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

pub enum RibKind {
    // No translation needs to be applied.
    NormalRibKind,

    // We passed through a function scope at the given node ID. Translate
    // upvars as appropriate.
    FunctionRibKind(node_id /* func id */, node_id /* body id */),

    // We passed through an impl or trait and are now in one of its
    // methods. Allow references to ty params that that impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
              // parent;   method itself
    MethodRibKind(node_id, MethodSort),

    // We passed through a function *item* scope. Disallow upvars.
    OpaqueFunctionRibKind,

    // We're in a constant item. Can't refer to dynamic stuff.
    ConstantItemRibKind
}

// Methods can be required or provided. Required methods only occur in traits.
pub enum MethodSort {
    Required,
    Provided(node_id)
}

// The X-ray flag indicates that a context has the X-ray privilege, which
// allows it to reference private names. Currently, this is used for the test
// runner.
//
// FIXME #4947: The X-ray flag is kind of questionable in the first
// place. It might be better to introduce an expr_xray_path instead.

#[deriving(Eq)]
pub enum XrayFlag {
    NoXray,     //< Private items cannot be accessed.
    Xray        //< Private items can be accessed.
}

pub enum UseLexicalScopeFlag {
    DontUseLexicalScope,
    UseLexicalScope
}

pub enum SearchThroughModulesFlag {
    DontSearchThroughModules,
    SearchThroughModules
}

pub enum ModulePrefixResult {
    NoPrefixFound,
    PrefixFound(@mut Module, uint)
}

#[deriving(Eq)]
pub enum AllowCapturingSelfFlag {
    AllowCapturingSelf,         //< The "self" definition can be captured.
    DontAllowCapturingSelf,     //< The "self" definition cannot be captured.
}

#[deriving(Eq)]
enum NameSearchType {
    SearchItemsAndPublicImports,    //< Search items and public imports.
    SearchItemsAndAllImports,       //< Search items and all imports.
}

pub enum BareIdentifierPatternResolution {
    FoundStructOrEnumVariant(def),
    FoundConst(def),
    BareIdentifierPatternUnresolved
}

// Specifies how duplicates should be handled when adding a child item if
// another item exists with the same name in some namespace.
#[deriving(Eq)]
pub enum DuplicateCheckingMode {
    ForbidDuplicateModules,
    ForbidDuplicateTypes,
    ForbidDuplicateValues,
    ForbidDuplicateTypesAndValues,
    OverwriteDuplicates
}

// Returns the namespace associated with the given duplicate checking mode,
// or fails for OverwriteDuplicates. This is used for error messages.
pub fn namespace_for_duplicate_checking_mode(mode: DuplicateCheckingMode)
                                          -> Namespace {
    match mode {
        ForbidDuplicateModules | ForbidDuplicateTypes |
        ForbidDuplicateTypesAndValues => TypeNS,
        ForbidDuplicateValues => ValueNS,
        OverwriteDuplicates => fail!(~"OverwriteDuplicates has no namespace")
    }
}

/// One local scope.
pub struct Rib {
    bindings: @mut HashMap<ident,def_like>,
    kind: RibKind,
}

pub fn Rib(kind: RibKind) -> Rib {
    Rib {
        bindings: @mut HashMap::new(),
        kind: kind
    }
}


/// One import directive.
pub struct ImportDirective {
    privacy: Privacy,
    module_path: ~[ident],
    subclass: @ImportDirectiveSubclass,
    span: span,
}

pub fn ImportDirective(privacy: Privacy,
                       module_path: ~[ident],
                       subclass: @ImportDirectiveSubclass,
                       span: span)
                    -> ImportDirective {
    ImportDirective {
        privacy: privacy,
        module_path: module_path,
        subclass: subclass,
        span: span
    }
}

/// The item that an import resolves to.
pub struct Target {
    target_module: @mut Module,
    bindings: @mut NameBindings,
}

pub fn Target(target_module: @mut Module,
              bindings: @mut NameBindings)
           -> Target {
    Target {
        target_module: target_module,
        bindings: bindings
    }
}

/// An ImportResolution represents a particular `use` directive.
pub struct ImportResolution {
    /// The privacy of this `use` directive (whether it's `use` or
    /// `pub use`.
    privacy: Privacy,
    span: span,

    // The number of outstanding references to this name. When this reaches
    // zero, outside modules can count on the targets being correct. Before
    // then, all bets are off; future imports could override this name.

    outstanding_references: uint,

    /// The value that this `use` directive names, if there is one.
    value_target: Option<Target>,
    /// The type that this `use` directive names, if there is one.
    type_target: Option<Target>,

    /// There exists one state per import statement
    state: @mut ImportState,
}

pub fn ImportResolution(privacy: Privacy,
                        span: span,
                        state: @mut ImportState) -> ImportResolution {
    ImportResolution {
        privacy: privacy,
        span: span,
        outstanding_references: 0,
        value_target: None,
        type_target: None,
        state: state,
    }
}

pub impl ImportResolution {
    fn target_for_namespace(&self, namespace: Namespace) -> Option<Target> {
        match namespace {
            TypeNS      => return copy self.type_target,
            ValueNS     => return copy self.value_target
        }
    }
}

pub struct ImportState {
    used: bool,
    warned: bool
}

pub fn ImportState() -> ImportState {
    ImportState{ used: false, warned: false }
}

/// The link from a module up to its nearest parent node.
pub enum ParentLink {
    NoParentLink,
    ModuleParentLink(@mut Module, ident),
    BlockParentLink(@mut Module, node_id)
}

/// The type of module this is.
pub enum ModuleKind {
    NormalModuleKind,
    ExternModuleKind,
    TraitModuleKind,
    AnonymousModuleKind,
}

/// One node in the tree of modules.
pub struct Module {
    parent_link: ParentLink,
    def_id: Option<def_id>,
    kind: ModuleKind,

    children: @mut HashMap<ident, @mut NameBindings>,
    imports: @mut ~[@ImportDirective],

    // The external module children of this node that were declared with
    // `extern mod`.
    external_module_children: @mut HashMap<ident, @mut Module>,

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

    anonymous_children: @mut HashMap<node_id,@mut Module>,

    // The status of resolving each import in this module.
    import_resolutions: @mut HashMap<ident, @mut ImportResolution>,

    // The number of unresolved globs that this module exports.
    glob_count: uint,

    // The index of the import we're resolving.
    resolved_import_count: uint,
}

pub fn Module(parent_link: ParentLink,
              def_id: Option<def_id>,
              kind: ModuleKind)
           -> Module {
    Module {
        parent_link: parent_link,
        def_id: def_id,
        kind: kind,
        children: @mut HashMap::new(),
        imports: @mut ~[],
        external_module_children: @mut HashMap::new(),
        anonymous_children: @mut HashMap::new(),
        import_resolutions: @mut HashMap::new(),
        glob_count: 0,
        resolved_import_count: 0
    }
}

pub impl Module {
    fn all_imports_resolved(&self) -> bool {
        let imports = &mut *self.imports;
        return imports.len() == self.resolved_import_count;
    }
}

// Records a possibly-private type definition.
pub struct TypeNsDef {
    privacy: Privacy,
    module_def: Option<@mut Module>,
    type_def: Option<def>
}

// Records a possibly-private value definition.
pub struct ValueNsDef {
    privacy: Privacy,
    def: def,
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
pub struct NameBindings {
    type_def: Option<TypeNsDef>,    //< Meaning in type namespace.
    value_def: Option<ValueNsDef>,  //< Meaning in value namespace.

    // For error reporting
    // FIXME (#3783): Merge me into TypeNsDef and ValueNsDef.
    type_span: Option<span>,
    value_span: Option<span>,
}

pub impl NameBindings {
    /// Creates a new module in this set of name bindings.
    fn define_module(@mut self,
                     privacy: Privacy,
                     parent_link: ParentLink,
                     def_id: Option<def_id>,
                     kind: ModuleKind,
                     sp: span) {
        // Merges the module with the existing type def or creates a new one.
        let module_ = @mut Module(parent_link, def_id, kind);
        match self.type_def {
            None => {
                self.type_def = Some(TypeNsDef {
                    privacy: privacy,
                    module_def: Some(module_),
                    type_def: None
                });
            }
            Some(copy type_def) => {
                self.type_def = Some(TypeNsDef {
                    privacy: privacy,
                    module_def: Some(module_),
                    .. type_def
                });
            }
        }
        self.type_span = Some(sp);
    }

    /// Records a type definition.
    fn define_type(@mut self, privacy: Privacy, def: def, sp: span) {
        // Merges the type with the existing type def or creates a new one.
        match self.type_def {
            None => {
                self.type_def = Some(TypeNsDef {
                    privacy: privacy,
                    module_def: None,
                    type_def: Some(def)
                });
            }
            Some(copy type_def) => {
                self.type_def = Some(TypeNsDef {
                    privacy: privacy,
                    type_def: Some(def),
                    .. type_def
                });
            }
        }
        self.type_span = Some(sp);
    }

    /// Records a value definition.
    fn define_value(@mut self, privacy: Privacy, def: def, sp: span) {
        self.value_def = Some(ValueNsDef { privacy: privacy, def: def });
        self.value_span = Some(sp);
    }

    /// Returns the module node if applicable.
    fn get_module_if_available(&self) -> Option<@mut Module> {
        match self.type_def {
            Some(ref type_def) => (*type_def).module_def,
            None => None
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module(@mut self) -> @mut Module {
        match self.get_module_if_available() {
            None => {
                fail!(~"get_module called on a node with no module \
                       definition!")
            }
            Some(module_def) => module_def
        }
    }

    fn defined_in_namespace(&self, namespace: Namespace) -> bool {
        match namespace {
            TypeNS   => return self.type_def.is_some(),
            ValueNS  => return self.value_def.is_some()
        }
    }

    fn defined_in_public_namespace(&self, namespace: Namespace) -> bool {
        match namespace {
            TypeNS => match self.type_def {
                Some(def) => def.privacy != Private,
                None => false
            },
            ValueNS => match self.value_def {
                Some(def) => def.privacy != Private,
                None => false
            }
        }
    }

    fn def_for_namespace(&self, namespace: Namespace) -> Option<def> {
        match namespace {
            TypeNS => {
                match self.type_def {
                    None => None,
                    Some(ref type_def) => {
                        // FIXME (#3784): This is reallllly questionable.
                        // Perhaps the right thing to do is to merge def_mod
                        // and def_ty.
                        match (*type_def).type_def {
                            Some(type_def) => Some(type_def),
                            None => {
                                match (*type_def).module_def {
                                    Some(module_def) => {
                                        let module_def = &mut *module_def;
                                        module_def.def_id.map(|def_id|
                                            def_mod(*def_id))
                                    }
                                    None => None
                                }
                            }
                        }
                    }
                }
            }
            ValueNS => {
                match self.value_def {
                    None => None,
                    Some(value_def) => Some(value_def.def)
                }
            }
        }
    }

    fn privacy_for_namespace(&self, namespace: Namespace) -> Option<Privacy> {
        match namespace {
            TypeNS => {
                match self.type_def {
                    None => None,
                    Some(ref type_def) => Some((*type_def).privacy)
                }
            }
            ValueNS => {
                match self.value_def {
                    None => None,
                    Some(value_def) => Some(value_def.privacy)
                }
            }
        }
    }

    fn span_for_namespace(&self, namespace: Namespace) -> Option<span> {
        if self.defined_in_namespace(namespace) {
            match namespace {
                TypeNS  => self.type_span,
                ValueNS => self.value_span,
            }
        } else {
            None
        }
    }
}

pub fn NameBindings() -> NameBindings {
    NameBindings {
        type_def: None,
        value_def: None,
        type_span: None,
        value_span: None
    }
}

/// Interns the names of the primitive types.
pub struct PrimitiveTypeTable {
    primitive_types: HashMap<ident,prim_ty>,
}

pub impl PrimitiveTypeTable {
    fn intern(&mut self, intr: @ident_interner, string: @~str,
              primitive_type: prim_ty) {
        let ident = intr.intern(string);
        self.primitive_types.insert(ident, primitive_type);
    }
}

pub fn PrimitiveTypeTable(intr: @ident_interner) -> PrimitiveTypeTable {
    let mut table = PrimitiveTypeTable {
        primitive_types: HashMap::new()
    };

    table.intern(intr, @~"bool",    ty_bool);
    table.intern(intr, @~"char",    ty_int(ty_char));
    table.intern(intr, @~"float",   ty_float(ty_f));
    table.intern(intr, @~"f32",     ty_float(ty_f32));
    table.intern(intr, @~"f64",     ty_float(ty_f64));
    table.intern(intr, @~"int",     ty_int(ty_i));
    table.intern(intr, @~"i8",      ty_int(ty_i8));
    table.intern(intr, @~"i16",     ty_int(ty_i16));
    table.intern(intr, @~"i32",     ty_int(ty_i32));
    table.intern(intr, @~"i64",     ty_int(ty_i64));
    table.intern(intr, @~"str",     ty_str);
    table.intern(intr, @~"uint",    ty_uint(ty_u));
    table.intern(intr, @~"u8",      ty_uint(ty_u8));
    table.intern(intr, @~"u16",     ty_uint(ty_u16));
    table.intern(intr, @~"u32",     ty_uint(ty_u32));
    table.intern(intr, @~"u64",     ty_uint(ty_u64));

    return table;
}


pub fn namespace_to_str(ns: Namespace) -> ~str {
    match ns {
        TypeNS  => ~"type",
        ValueNS => ~"value",
    }
}

pub fn Resolver(session: Session,
                lang_items: LanguageItems,
                crate: @crate)
             -> Resolver {
    let graph_root = @mut NameBindings();

    graph_root.define_module(Public,
                             NoParentLink,
                             Some(def_id { crate: 0, node: 0 }),
                             NormalModuleKind,
                             crate.span);

    let current_module = graph_root.get_module();

    let self = Resolver {
        session: @session,
        lang_items: copy lang_items,
        crate: crate,

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        graph_root: graph_root,

        trait_info: HashMap::new(),
        structs: HashSet::new(),

        unresolved_imports: 0,

        current_module: current_module,
        value_ribs: ~[],
        type_ribs: ~[],
        label_ribs: ~[],

        xray_context: NoXray,
        current_trait_refs: None,

        self_ident: special_idents::self_,
        type_self_ident: special_idents::type_self,

        primitive_type_table: @PrimitiveTypeTable(session.
                                                  parse_sess.interner),

        namespaces: ~[ TypeNS, ValueNS ],

        attr_main_fn: None,
        main_fns: ~[],

        start_fn: None,

        def_map: @mut HashMap::new(),
        export_map2: @mut HashMap::new(),
        trait_map: HashMap::new(),

        intr: session.intr()
    };

    self
}

/// The main resolver class.
pub struct Resolver {
    session: @Session,
    lang_items: LanguageItems,
    crate: @crate,

    intr: @ident_interner,

    graph_root: @mut NameBindings,

    trait_info: HashMap<def_id, HashSet<ident>>,
    structs: HashSet<def_id>,

    // The number of imports that are currently unresolved.
    unresolved_imports: uint,

    // The module that represents the current item scope.
    current_module: @mut Module,

    // The current set of local scopes, for values.
    // FIXME #4948: Reuse ribs to avoid allocation.
    value_ribs: ~[@Rib],

    // The current set of local scopes, for types.
    type_ribs: ~[@Rib],

    // The current set of local scopes, for labels.
    label_ribs: ~[@Rib],

    // Whether the current context is an X-ray context. An X-ray context is
    // allowed to access private names of any module.
    xray_context: XrayFlag,

    // The trait that the current context can refer to.
    current_trait_refs: Option<~[def_id]>,

    // The ident for the keyword "self".
    self_ident: ident,
    // The ident for the non-keyword "Self".
    type_self_ident: ident,

    // The idents for the primitive types.
    primitive_type_table: @PrimitiveTypeTable,

    // The four namespaces.
    namespaces: ~[Namespace],

    // The function that has attribute named 'main'
    attr_main_fn: Option<(node_id, span)>,

    // The functions that could be main functions
    main_fns: ~[Option<(node_id, span)>],

    // The function that has the attribute 'start' on it
    start_fn: Option<(node_id, span)>,

    def_map: DefMap,
    export_map2: ExportMap2,
    trait_map: TraitMap,
}

pub impl Resolver {
    /// The main name resolution procedure.
    fn resolve(@mut self) {
        self.build_reduced_graph();
        self.session.abort_if_errors();

        self.resolve_imports();
        self.session.abort_if_errors();

        self.record_exports();
        self.session.abort_if_errors();

        self.resolve_crate();
        self.session.abort_if_errors();

        self.check_duplicate_main();
        self.check_for_unused_imports_if_necessary();
    }

    //
    // Reduced graph building
    //
    // Here we build the "reduced graph": the graph of the module tree without
    // any imports resolved.
    //

    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(@mut self) {
        let initial_parent =
            ModuleReducedGraphParent(self.graph_root.get_module());
        visit_crate(self.crate, initial_parent, mk_vt(@Visitor {
            visit_item: |item, context, visitor|
                self.build_reduced_graph_for_item(item, context, visitor),

            visit_foreign_item: |foreign_item, context, visitor|
                self.build_reduced_graph_for_foreign_item(foreign_item,
                                                             context,
                                                             visitor),

            visit_view_item: |view_item, context, visitor|
                self.build_reduced_graph_for_view_item(view_item,
                                                          context,
                                                          visitor),

            visit_block: |block, context, visitor|
                self.build_reduced_graph_for_block(block,
                                                      context,
                                                      visitor),

            .. *default_visitor()
        }));
    }

    /// Returns the current module tracked by the reduced graph parent.
    fn get_module_from_parent(@mut self,
                              reduced_graph_parent: ReducedGraphParent)
                           -> @mut Module {
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
    fn add_child(@mut self,
                 name: ident,
                 reduced_graph_parent: ReducedGraphParent,
                 duplicate_checking_mode: DuplicateCheckingMode,
                 // For printing errors
                 sp: span)
              -> (@mut NameBindings, ReducedGraphParent) {

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
        match module_.children.find(&name) {
            None => {
                let child = @mut NameBindings();
                module_.children.insert(name, child);
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

                let mut is_duplicate = false;
                match duplicate_checking_mode {
                    ForbidDuplicateModules => {
                        is_duplicate =
                            child.get_module_if_available().is_some();
                    }
                    ForbidDuplicateTypes => {
                        match child.def_for_namespace(TypeNS) {
                            Some(def_mod(_)) | None => {}
                            Some(_) => is_duplicate = true
                        }
                    }
                    ForbidDuplicateValues => {
                        is_duplicate = child.defined_in_namespace(ValueNS);
                    }
                    ForbidDuplicateTypesAndValues => {
                        match child.def_for_namespace(TypeNS) {
                            Some(def_mod(_)) | None => {}
                            Some(_) => is_duplicate = true
                        };
                        if child.defined_in_namespace(ValueNS) {
                            is_duplicate = true;
                        }
                    }
                    OverwriteDuplicates => {}
                }
                if duplicate_checking_mode != OverwriteDuplicates &&
                        is_duplicate {
                    // Return an error here by looking up the namespace that
                    // had the duplicate.
                    let ns = namespace_for_duplicate_checking_mode(
                        duplicate_checking_mode);
                    self.session.span_err(sp,
                        fmt!("duplicate definition of %s %s",
                             namespace_to_str(ns),
                             *self.session.str_of(name)));
                    for child.span_for_namespace(ns).each |sp| {
                        self.session.span_note(*sp,
                             fmt!("first definition of %s %s here:",
                                  namespace_to_str(ns),
                                  *self.session.str_of(name)));
                    }
                }
                return (*child, new_parent);
            }
        }
    }

    fn block_needs_anonymous_module(@mut self, block: &blk) -> bool {
        // If the block has view items, we need an anonymous module.
        if block.node.view_items.len() > 0 {
            return true;
        }

        // Check each statement.
        for block.node.stmts.each |statement| {
            match statement.node {
                stmt_decl(declaration, _) => {
                    match declaration.node {
                        decl_item(_) => {
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

    fn get_parent_link(@mut self,
                       parent: ReducedGraphParent,
                       name: ident)
                    -> ParentLink {
        match parent {
            ModuleReducedGraphParent(module_) => {
                return ModuleParentLink(module_, name);
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(@mut self,
                                    item: @item,
                                    parent: ReducedGraphParent,
                                    visitor: vt<ReducedGraphParent>) {
        let ident = item.ident;
        let sp = item.span;
        let privacy = visibility_to_privacy(item.vis);

        match item.node {
            item_mod(ref module_) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateModules, sp);

                let parent_link = self.get_parent_link(new_parent, ident);
                let def_id = def_id { crate: 0, node: item.id };
                name_bindings.define_module(privacy,
                                            parent_link,
                                            Some(def_id),
                                            NormalModuleKind,
                                            sp);

                let new_parent =
                    ModuleReducedGraphParent(name_bindings.get_module());

                visit_mod(module_, sp, item.id, new_parent, visitor);
            }

            item_foreign_mod(ref fm) => {
                let new_parent = match fm.sort {
                    named => {
                        let (name_bindings, new_parent) =
                            self.add_child(ident, parent,
                                           ForbidDuplicateModules, sp);

                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        let def_id = def_id { crate: 0, node: item.id };
                        name_bindings.define_module(privacy,
                                                    parent_link,
                                                    Some(def_id),
                                                    ExternModuleKind,
                                                    sp);

                        ModuleReducedGraphParent(name_bindings.get_module())
                    }

                    // For anon foreign mods, the contents just go in the
                    // current scope
                    anonymous => parent
                };

                visit_item(item, new_parent, visitor);
            }

            // These items live in the value namespace.
            item_const(*) => {
                let (name_bindings, _) =
                    self.add_child(ident, parent, ForbidDuplicateValues, sp);

                name_bindings.define_value
                    (privacy, def_const(local_def(item.id)), sp);
            }
            item_fn(_, purity, _, _, _) => {
              let (name_bindings, new_parent) =
                self.add_child(ident, parent, ForbidDuplicateValues, sp);

                let def = def_fn(local_def(item.id), purity);
                name_bindings.define_value(privacy, def, sp);
                visit_item(item, new_parent, visitor);
            }

            // These items live in the type namespace.
            item_ty(*) => {
                let (name_bindings, _) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (privacy, def_ty(local_def(item.id)), sp);
            }

            item_enum(ref enum_definition, _) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                name_bindings.define_type
                    (privacy, def_ty(local_def(item.id)), sp);

                for (*enum_definition).variants.each |variant| {
                    self.build_reduced_graph_for_variant(variant,
                        local_def(item.id),
                        // inherited => privacy of the enum item
                        variant_visibility_to_privacy(variant.node.vis,
                                                      privacy == Public),
                        new_parent,
                        visitor);
                }
            }

            // These items live in both the type and value namespaces.
            item_struct(struct_def, _) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                name_bindings.define_type(
                    privacy, def_ty(local_def(item.id)), sp);

                // If this struct is tuple-like or enum-like, define a name
                // in the value namespace.
                match struct_def.ctor_id {
                    None => {}
                    Some(ctor_id) => {
                        name_bindings.define_value(
                            privacy,
                            def_struct(local_def(ctor_id)),
                            sp);
                    }
                }

                // Record the def ID of this struct.
                self.structs.insert(local_def(item.id));

                visit_item(item, new_parent, visitor);
            }

            item_impl(_, trait_ref_opt, ty, ref methods) => {
                // If this implements an anonymous trait and it has static
                // methods, then add all the static methods within to a new
                // module, if the type was defined within this module.
                //
                // FIXME (#3785): This is quite unsatisfactory. Perhaps we
                // should modify anonymous traits to only be implementable in
                // the same module that declared the type.

                // Bail out early if there are no static methods.
                let mut has_static_methods = false;
                for methods.each |method| {
                    match method.self_ty.node {
                        sty_static => has_static_methods = true,
                        _ => {}
                    }
                }

                // If there are static methods, then create the module
                // and add them.
                match (trait_ref_opt, ty) {
                    (None, @Ty { node: ty_path(path, _), _ }) if
                            has_static_methods && path.idents.len() == 1 => {
                        // Create the module.
                        let name = path_to_ident(path);
                        let (name_bindings, new_parent) =
                            self.add_child(name,
                                           parent,
                                           ForbidDuplicateModules,
                                           sp);

                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        let def_id = local_def(item.id);
                        name_bindings.define_module(Public,
                                                    parent_link,
                                                    Some(def_id),
                                                    TraitModuleKind,
                                                    sp);

                        let new_parent = ModuleReducedGraphParent(
                            name_bindings.get_module());

                        // For each static method...
                        for methods.each |method| {
                            match method.self_ty.node {
                                sty_static => {
                                    // Add the static method to the
                                    // module.
                                    let ident = method.ident;
                                    let (method_name_bindings, _) =
                                        self.add_child(
                                            ident,
                                            new_parent,
                                            ForbidDuplicateValues,
                                            method.span);
                                    let def = def_fn(local_def(method.id),
                                                     method.purity);
                                    method_name_bindings.define_value(
                                        Public, def, method.span);
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {}
                }

                visit_item(item, parent, visitor);
            }

            item_trait(_, _, ref methods) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                // If the trait has static methods, then add all the static
                // methods within to a new module.
                //
                // We only need to create the module if the trait has static
                // methods, so check that first.
                let mut has_static_methods = false;
                for (*methods).each |method| {
                    let ty_m = trait_method_to_ty_method(method);
                    match ty_m.self_ty.node {
                        sty_static => {
                            has_static_methods = true;
                            break;
                        }
                        _ => {}
                    }
                }

                // Create the module if necessary.
                let module_parent_opt;
                if has_static_methods {
                    let parent_link = self.get_parent_link(parent, ident);
                    name_bindings.define_module(privacy,
                                                parent_link,
                                                Some(local_def(item.id)),
                                                TraitModuleKind,
                                                sp);
                    module_parent_opt = Some(ModuleReducedGraphParent(
                        name_bindings.get_module()));
                } else {
                    module_parent_opt = None;
                }

                // Add the names of all the methods to the trait info.
                let mut method_names = HashSet::new();
                for methods.each |method| {
                    let ty_m = trait_method_to_ty_method(method);

                    let ident = ty_m.ident;
                    // Add it to the trait info if not static,
                    // add it as a name in the trait module otherwise.
                    match ty_m.self_ty.node {
                        sty_static => {
                            let def = def_static_method(
                                local_def(ty_m.id),
                                Some(local_def(item.id)),
                                ty_m.purity);

                            let (method_name_bindings, _) =
                                self.add_child(ident,
                                               module_parent_opt.get(),
                                               ForbidDuplicateValues,
                                               ty_m.span);
                            method_name_bindings.define_value(Public,
                                                              def,
                                                              ty_m.span);
                        }
                        _ => {
                            method_names.insert(ident);
                        }
                    }
                }

                let def_id = local_def(item.id);
                self.trait_info.insert(def_id, method_names);

                name_bindings.define_type(privacy, def_trait(def_id), sp);
                visit_item(item, new_parent, visitor);
            }

            item_mac(*) => {
                fail!(~"item macros unimplemented")
            }
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and/or value namespaces.
    fn build_reduced_graph_for_variant(@mut self,
                                       variant: &variant,
                                       item_id: def_id,
                                       parent_privacy: Privacy,
                                       parent: ReducedGraphParent,
                                       _visitor: vt<ReducedGraphParent>) {
        let ident = variant.node.name;
        let (child, _) = self.add_child(ident, parent, ForbidDuplicateValues,
                                        variant.span);

        let privacy;
        match variant.node.vis {
            public => privacy = Public,
            private => privacy = Private,
            inherited => privacy = parent_privacy
        }

        match variant.node.kind {
            tuple_variant_kind(_) => {
                child.define_value(privacy,
                                   def_variant(item_id,
                                               local_def(variant.node.id)),
                                   variant.span);
            }
            struct_variant_kind(_) => {
                child.define_type(privacy,
                                  def_variant(item_id,
                                              local_def(variant.node.id)),
                                  variant.span);
                self.structs.insert(local_def(variant.node.id));
            }
        }
    }

    /**
     * Constructs the reduced graph for one 'view item'. View items consist
     * of imports and use directives.
     */
    fn build_reduced_graph_for_view_item(@mut self,
                                         view_item: @view_item,
                                         parent: ReducedGraphParent,
                                         _visitor: vt<ReducedGraphParent>) {
        let privacy = visibility_to_privacy(view_item.vis);
        match view_item.node {
            view_item_use(ref view_paths) => {
                for view_paths.each |view_path| {
                    // Extract and intern the module part of the path. For
                    // globs and lists, the path is found directly in the AST;
                    // for simple paths we have to munge the path a little.

                    let mut module_path = ~[];
                    match view_path.node {
                        view_path_simple(_, full_path, _) => {
                            let path_len = full_path.idents.len();
                            assert!(path_len != 0);

                            for full_path.idents.eachi |i, ident| {
                                if i != path_len - 1 {
                                    module_path.push(*ident);
                                }
                            }
                        }

                        view_path_glob(module_ident_path, _) |
                        view_path_list(module_ident_path, _, _) => {
                            for module_ident_path.idents.each |ident| {
                                module_path.push(*ident);
                            }
                        }
                    }

                    // Build up the import directives.
                    let module_ = self.get_module_from_parent(parent);
                    match view_path.node {
                        view_path_simple(binding, full_path, _) => {
                            let source_ident = *full_path.idents.last();
                            let subclass = @SingleImport(binding,
                                                         source_ident);
                            self.build_import_directive(privacy,
                                                        module_,
                                                        module_path,
                                                        subclass,
                                                        view_path.span);
                        }
                        view_path_list(_, ref source_idents, _) => {
                            for source_idents.each |source_ident| {
                                let name = source_ident.node.name;
                                let subclass = @SingleImport(name, name);
                                self.build_import_directive(privacy,
                                                            module_,
                                                            copy module_path,
                                                            subclass,
                                                            source_ident.span);
                            }
                        }
                        view_path_glob(_, _) => {
                            self.build_import_directive(privacy,
                                                        module_,
                                                        module_path,
                                                        @GlobImport,
                                                        view_path.span);
                        }
                    }
                }
            }

            view_item_extern_mod(name, _, node_id) => {
                match find_extern_mod_stmt_cnum(self.session.cstore,
                                                node_id) {
                    Some(crate_id) => {
                        let def_id = def_id { crate: crate_id, node: 0 };
                        let parent_link = ModuleParentLink
                            (self.get_module_from_parent(parent), name);
                        let external_module = @mut Module(parent_link,
                                                          Some(def_id),
                                                          NormalModuleKind);

                        parent.external_module_children.insert(
                            name,
                            external_module);

                        self.build_reduced_graph_for_external_crate(
                            external_module);
                    }
                    None => {}  // Ignore.
                }
            }
        }
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(@mut self,
                                            foreign_item: @foreign_item,
                                            parent: ReducedGraphParent,
                                            visitor:
                                                vt<ReducedGraphParent>) {
        let name = foreign_item.ident;
        let (name_bindings, new_parent) =
            self.add_child(name, parent, ForbidDuplicateValues,
                           foreign_item.span);

        match foreign_item.node {
            foreign_item_fn(_, _, ref generics) => {
                let def = def_fn(local_def(foreign_item.id), unsafe_fn);
                name_bindings.define_value(Public, def, foreign_item.span);

                do self.with_type_parameter_rib(
                    HasTypeParameters(
                        generics, foreign_item.id, 0, NormalRibKind))
                {
                    visit_foreign_item(foreign_item, new_parent, visitor);
                }
            }
            foreign_item_const(*) => {
                let def = def_const(local_def(foreign_item.id));
                name_bindings.define_value(Public, def, foreign_item.span);

                visit_foreign_item(foreign_item, new_parent, visitor);
            }
        }
    }

    fn build_reduced_graph_for_block(@mut self,
                                     block: &blk,
                                     parent: ReducedGraphParent,
                                     visitor: vt<ReducedGraphParent>) {
        let new_parent;
        if self.block_needs_anonymous_module(block) {
            let block_id = block.node.id;

            debug!("(building reduced graph for block) creating a new \
                    anonymous module for block %d",
                   block_id);

            let parent_module = self.get_module_from_parent(parent);
            let new_module = @mut Module(
                BlockParentLink(parent_module, block_id),
                None,
                AnonymousModuleKind);
            parent_module.anonymous_children.insert(block_id, new_module);
            new_parent = ModuleReducedGraphParent(new_module);
        } else {
            new_parent = parent;
        }

        visit_block(block, new_parent, visitor);
    }

    fn handle_external_def(@mut self,
                           def: def,
                           modules: &mut HashMap<def_id, @mut Module>,
                           child_name_bindings: @mut NameBindings,
                           final_ident: &str,
                           ident: ident,
                           new_parent: ReducedGraphParent) {
        match def {
          def_mod(def_id) | def_foreign_mod(def_id) => {
            match child_name_bindings.type_def {
              Some(TypeNsDef { module_def: Some(copy module_def), _ }) => {
                debug!("(building reduced graph for external crate) \
                        already created module");
                module_def.def_id = Some(def_id);
                modules.insert(def_id, module_def);
              }
              Some(_) | None => {
                debug!("(building reduced graph for \
                        external crate) building module \
                        %s", final_ident);
                let parent_link = self.get_parent_link(new_parent, ident);

                // FIXME (#5074): this should be a match on find
                if !modules.contains_key(&def_id) {
                    child_name_bindings.define_module(Public,
                                                      parent_link,
                                                      Some(def_id),
                                                      NormalModuleKind,
                                                      dummy_sp());
                    modules.insert(def_id,
                                   child_name_bindings.get_module());
                } else {
                    let existing_module = *modules.get(&def_id);
                    // Create an import resolution to
                    // avoid creating cycles in the
                    // module graph.

                    let resolution =
                        @mut ImportResolution(Public,
                                              dummy_sp(),
                                              @mut ImportState());
                    resolution.outstanding_references = 0;

                    match existing_module.parent_link {
                      NoParentLink |
                      BlockParentLink(*) => {
                        fail!(~"can't happen");
                      }
                      ModuleParentLink(parent_module, ident) => {
                        let name_bindings = parent_module.children.get(
                            &ident);
                        resolution.type_target =
                            Some(Target(parent_module, *name_bindings));
                      }
                    }

                    debug!("(building reduced graph for external crate) \
                            ... creating import resolution");

                    new_parent.import_resolutions.insert(ident, resolution);
                }
              }
            }
          }
          def_fn(*) | def_static_method(*) | def_const(*) |
          def_variant(*) => {
            debug!("(building reduced graph for external \
                    crate) building value %s", final_ident);
            child_name_bindings.define_value(Public, def, dummy_sp());
          }
          def_trait(def_id) => {
              debug!("(building reduced graph for external \
                      crate) building type %s", final_ident);

              // If this is a trait, add all the method names
              // to the trait info.

              let method_def_ids = get_trait_method_def_ids(self.session.cstore,
                                                            def_id);
              let mut interned_method_names = HashSet::new();
              for method_def_ids.each |&method_def_id| {
                  let (method_name, self_ty) =
                      get_method_name_and_self_ty(self.session.cstore,
                                                  method_def_id);

                  debug!("(building reduced graph for \
                          external crate) ... adding \
                          trait method '%s'",
                         *self.session.str_of(method_name));

                  // Add it to the trait info if not static.
                  if self_ty != sty_static {
                      interned_method_names.insert(method_name);
                  }
              }
              self.trait_info.insert(def_id, interned_method_names);

              child_name_bindings.define_type(Public, def, dummy_sp());
          }
          def_ty(_) => {
              debug!("(building reduced graph for external \
                      crate) building type %s", final_ident);

              child_name_bindings.define_type(Public, def, dummy_sp());
          }
          def_struct(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type %s",
                   final_ident);
            child_name_bindings.define_type(Public, def, dummy_sp());
            self.structs.insert(def_id);
          }
          def_self(*) | def_arg(*) | def_local(*) |
          def_prim_ty(*) | def_ty_param(*) | def_binding(*) |
          def_use(*) | def_upvar(*) | def_region(*) |
          def_typaram_binder(*) | def_label(*) | def_self_ty(*) => {
            fail!(fmt!("didn't expect `%?`", def));
          }
        }
    }

    /**
     * Builds the reduced graph rooted at the 'use' directive for an external
     * crate.
     */
    fn build_reduced_graph_for_external_crate(@mut self, root: @mut Module) {
        let mut modules = HashMap::new();

        // Create all the items reachable by paths.
        for each_path(self.session.cstore, root.def_id.get().crate)
                |path_string, def_like| {

            debug!("(building reduced graph for external crate) found path \
                        entry: %s (%?)",
                    path_string, def_like);

            let mut pieces = ~[];
            for each_split_str(path_string, "::") |s| { pieces.push(s.to_owned()) }
            let final_ident_str = pieces.pop();
            let final_ident = self.session.ident_of(final_ident_str);

            // Find the module we need, creating modules along the way if we
            // need to.

            let mut current_module = root;
            for pieces.each |ident_str| {
                let ident = self.session.ident_of(/*bad*/copy *ident_str);
                // Create or reuse a graph node for the child.
                let (child_name_bindings, new_parent) =
                    self.add_child(ident,
                                   ModuleReducedGraphParent(current_module),
                                   OverwriteDuplicates,
                                   dummy_sp());

                // Define or reuse the module node.
                match child_name_bindings.type_def {
                    None => {
                        debug!("(building reduced graph for external crate) \
                                autovivifying missing type def %s",
                                *ident_str);
                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        child_name_bindings.define_module(Public,
                                                          parent_link,
                                                          None,
                                                          NormalModuleKind,
                                                          dummy_sp());
                    }
                    Some(copy type_ns_def)
                            if type_ns_def.module_def.is_none() => {
                        debug!("(building reduced graph for external crate) \
                                autovivifying missing module def %s",
                                *ident_str);
                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        child_name_bindings.define_module(Public,
                                                          parent_link,
                                                          None,
                                                          NormalModuleKind,
                                                          dummy_sp());
                    }
                    _ => {} // Fall through.
                }

                current_module = child_name_bindings.get_module();
            }

            match def_like {
                dl_def(def) => {
                    // Add the new child item.
                    let (child_name_bindings, new_parent) =
                        self.add_child(final_ident,
                                       ModuleReducedGraphParent(
                                            current_module),
                                       OverwriteDuplicates,
                                       dummy_sp());

                    self.handle_external_def(def,
                                             &mut modules,
                                             child_name_bindings,
                                             *self.session.str_of(
                                                 final_ident),
                                             final_ident,
                                             new_parent);
                }
                dl_impl(def) => {
                    // We only process static methods of impls here.
                    match get_type_name_if_impl(self.session.cstore, def) {
                        None => {}
                        Some(final_ident) => {
                            let static_methods_opt =
                                get_static_methods_if_impl(
                                    self.session.cstore, def);
                            match static_methods_opt {
                                Some(ref static_methods) if
                                    static_methods.len() >= 1 => {
                                    debug!("(building reduced graph for \
                                            external crate) processing \
                                            static methods for type name %s",
                                            *self.session.str_of(
                                                final_ident));

                                    let (child_name_bindings, new_parent) =
                                        self.add_child(final_ident,
                                            ModuleReducedGraphParent(
                                                            current_module),
                                            OverwriteDuplicates,
                                            dummy_sp());

                                    // Process the static methods. First,
                                    // create the module.
                                    let type_module;
                                    match child_name_bindings.type_def {
                                        Some(TypeNsDef {
                                            module_def: Some(copy module_def),
                                            _
                                        }) => {
                                            // We already have a module. This
                                            // is OK.
                                            type_module = module_def;
                                        }
                                        Some(_) | None => {
                                            let parent_link =
                                                self.get_parent_link(
                                                    new_parent, final_ident);
                                            child_name_bindings.define_module(
                                                Public,
                                                parent_link,
                                                Some(def),
                                                NormalModuleKind,
                                                dummy_sp());
                                            type_module =
                                                child_name_bindings.
                                                    get_module();
                                        }
                                    }

                                    // Add each static method to the module.
                                    let new_parent = ModuleReducedGraphParent(
                                        type_module);
                                    for static_methods.each
                                            |static_method_info| {
                                        let ident = static_method_info.ident;
                                        debug!("(building reduced graph for \
                                                 external crate) creating \
                                                 static method '%s'",
                                               *self.session.str_of(ident));

                                        let (method_name_bindings, _) =
                                            self.add_child(
                                                ident,
                                                new_parent,
                                                OverwriteDuplicates,
                                                dummy_sp());
                                        let def = def_fn(
                                            static_method_info.def_id,
                                            static_method_info.purity);
                                        method_name_bindings.define_value(
                                            Public, def, dummy_sp());
                                    }
                                }

                                // Otherwise, do nothing.
                                Some(_) | None => {}
                            }
                        }
                    }
                }
                dl_field => {
                    debug!("(building reduced graph for external crate) \
                            ignoring field");
                }
            }
        }
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(@mut self,
                              privacy: Privacy,
                              module_: @mut Module,
                              module_path: ~[ident],
                              subclass: @ImportDirectiveSubclass,
                              span: span) {
        let directive = @ImportDirective(privacy, module_path,
                                         subclass, span);
        module_.imports.push(directive);

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match *subclass {
            SingleImport(target, _) => {
                debug!("(building import directive) building import \
                        directive: privacy %? %s::%s",
                       privacy,
                       self.idents_to_str(directive.module_path),
                       *self.session.str_of(target));

                match module_.import_resolutions.find(&target) {
                    Some(resolution) => {
                        debug!("(building import directive) bumping \
                                reference");
                        resolution.outstanding_references += 1;
                    }
                    None => {
                        debug!("(building import directive) creating new");
                        let state = @mut ImportState();
                        let resolution = @mut ImportResolution(privacy,
                                                               span,
                                                               state);
                        let name = self.idents_to_str(directive.module_path);
                        // Don't warn about unused intrinsics because they're
                        // automatically appended to all files
                        if name == ~"intrinsic::rusti" {
                            resolution.state.warned = true;
                        }
                        resolution.outstanding_references = 1;
                        module_.import_resolutions.insert(target, resolution);
                    }
                }
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.glob_count += 1;
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

    /**
     * Resolves all imports for the crate. This method performs the fixed-
     * point iteration.
     */
    fn resolve_imports(@mut self) {
        let mut i = 0;
        let mut prev_unresolved_imports = 0;
        loop {
            debug!("(resolving imports) iteration %u, %u imports left",
                   i, self.unresolved_imports);

            let module_root = self.graph_root.get_module();
            self.resolve_imports_for_module_subtree(module_root);

            if self.unresolved_imports == 0 {
                debug!("(resolving imports) success");
                break;
            }

            if self.unresolved_imports == prev_unresolved_imports {
                self.session.err(~"failed to resolve imports");
                self.report_unresolved_imports(module_root);
                break;
            }

            i += 1;
            prev_unresolved_imports = self.unresolved_imports;
        }
    }

    /// Attempts to resolve imports for the given module and all of its
    /// submodules.
    fn resolve_imports_for_module_subtree(@mut self, module_: @mut Module) {
        debug!("(resolving imports for module subtree) resolving %s",
               self.module_to_str(module_));
        self.resolve_imports_for_module(module_);

        for module_.children.each_value |&child_node| {
            match child_node.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.resolve_imports_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each_value |&child_module| {
            self.resolve_imports_for_module_subtree(child_module);
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(@mut self, module: @mut Module) {
        if module.all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for \
                   %s",
                   self.module_to_str(module));
            return;
        }

        let imports = &mut *module.imports;
        let import_count = imports.len();
        while module.resolved_import_count < import_count {
            let import_index = module.resolved_import_count;
            let import_directive = imports[import_index];
            match self.resolve_import_for_module(module, import_directive) {
                Failed => {
                    // We presumably emitted an error. Continue.
                    let msg = fmt!("failed to resolve import: %s",
                                   *self.import_path_to_str(
                                       import_directive.module_path,
                                       *import_directive.subclass));
                    self.session.span_err(import_directive.span, msg);
                }
                Indeterminate => {
                    // Bail out. We'll come around next time.
                    break;
                }
                Success(()) => {
                    // Good. Continue.
                }
            }

            module.resolved_import_count += 1;
        }
    }

    fn idents_to_str(@mut self, idents: &[ident]) -> ~str {
        let mut first = true;
        let mut result = ~"";
        for idents.each |ident| {
            if first { first = false; } else { result += "::" };
            result += *self.session.str_of(*ident);
        };
        return result;
    }

    fn import_directive_subclass_to_str(@mut self,
                                        subclass: ImportDirectiveSubclass)
                                     -> @~str {
        match subclass {
            SingleImport(_target, source) => self.session.str_of(source),
            GlobImport => @~"*"
        }
    }

    fn import_path_to_str(@mut self,
                          idents: &[ident],
                          subclass: ImportDirectiveSubclass)
                       -> @~str {
        if idents.is_empty() {
            self.import_directive_subclass_to_str(subclass)
        } else {
            @fmt!("%s::%s",
                 self.idents_to_str(idents),
                 *self.import_directive_subclass_to_str(subclass))
        }
    }

    /// Attempts to resolve the given import. The return value indicates
    /// failure if we're certain the name does not exist, indeterminate if we
    /// don't know whether the name exists at the moment due to other
    /// currently-unresolved imports, or success if we know the name exists.
    /// If successful, the resolved bindings are written into the module.
    fn resolve_import_for_module(@mut self, module_: @mut Module,
                                 import_directive: @ImportDirective)
                              -> ResolveResult<()> {
        let mut resolution_result = Failed;
        let module_path = &import_directive.module_path;

        debug!("(resolving import for module) resolving import `%s::...` in \
                `%s`",
               self.idents_to_str(*module_path),
               self.module_to_str(module_));

        // First, resolve the module path for the directive, if necessary.
        let containing_module = if module_path.len() == 0 {
            // Use the crate root.
            Some(self.graph_root.get_module())
        } else {
            match self.resolve_module_path_for_import(module_,
                                                      *module_path,
                                                      DontUseLexicalScope,
                                                      import_directive.span) {

                Failed => None,
                Indeterminate => {
                    resolution_result = Indeterminate;
                    None
                }
                Success(containing_module) => Some(containing_module),
            }
        };

        match containing_module {
            None => {}
            Some(containing_module) => {
                // We found the module that the target is contained
                // within. Attempt to resolve the import within it.

                match *import_directive.subclass {
                    SingleImport(target, source) => {
                        resolution_result =
                            self.resolve_single_import(module_,
                                                       containing_module,
                                                       target,
                                                       source);
                    }
                    GlobImport => {
                        let span = import_directive.span;
                        let privacy = import_directive.privacy;
                        resolution_result =
                            self.resolve_glob_import(privacy,
                                                     module_,
                                                     containing_module,
                                                     span);
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
                    assert!(module_.glob_count >= 1);
                    module_.glob_count -= 1;
                }
                SingleImport(*) => {
                    // Ignore.
                }
            }
        }

        return resolution_result;
    }

    fn create_name_bindings_from_module(module: @mut Module) -> NameBindings {
        NameBindings {
            type_def: Some(TypeNsDef {
                privacy: Public,
                module_def: Some(module),
                type_def: None,
            }),
            value_def: None,
            type_span: None,
            value_span: None,
        }
    }

    fn resolve_single_import(@mut self,
                             module_: @mut Module,
                             containing_module: @mut Module,
                             target: ident,
                             source: ident)
                          -> ResolveResult<()> {
        debug!("(resolving single import) resolving `%s` = `%s::%s` from \
                `%s`",
               *self.session.str_of(target),
               self.module_to_str(containing_module),
               *self.session.str_of(source),
               self.module_to_str(module_));

        // We need to resolve both namespaces for this to succeed.
        //
        // FIXME #4949: See if there's some way of handling namespaces in
        // a more generic way. We have two of them; it seems worth
        // doing...

        let mut value_result = UnknownResult;
        let mut type_result = UnknownResult;

        // Search for direct children of the containing module.
        match containing_module.children.find(&source) {
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

        // Unless we managed to find a result in both namespaces (unlikely),
        // search imports as well.
        match (value_result, type_result) {
            (BoundResult(*), BoundResult(*)) => {
                // Continue.
            }
            _ => {
                // If there is an unresolved glob at this point in the
                // containing module, bail out. We don't know enough to be
                // able to resolve this import.

                if containing_module.glob_count > 0 {
                    debug!("(resolving single import) unresolved glob; \
                            bailing out");
                    return Indeterminate;
                }

                // Now search the exported imports within the containing
                // module.

                match containing_module.import_resolutions.find(&source) {
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
                            if import_resolution.outstanding_references
                                == 0 => {

                        fn get_binding(import_resolution:
                                          @mut ImportResolution,
                                       namespace: Namespace)
                                    -> NamespaceResult {

                            // Import resolutions must be declared with "pub"
                            // in order to be exported.
                            if import_resolution.privacy == Private {
                                return UnboundResult;
                            }

                            match (*import_resolution).
                                    target_for_namespace(namespace) {
                                None => {
                                    return UnboundResult;
                                }
                                Some(target) => {
                                    import_resolution.state.used = true;
                                    return BoundResult(target.target_module,
                                                    target.bindings);
                                }
                            }
                        }

                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.
                        if value_result.is_unknown() {
                            value_result = get_binding(*import_resolution,
                                                       ValueNS);
                        }
                        if type_result.is_unknown() {
                            type_result = get_binding(*import_resolution,
                                                      TypeNS);
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
        match type_result {
            BoundResult(*) => {}
            _ => {
                match containing_module.external_module_children
                                       .find(&source) {
                    None => {} // Continue.
                    Some(module) => {
                        let name_bindings =
                            @mut Resolver::create_name_bindings_from_module(
                                *module);
                        type_result = BoundResult(containing_module,
                                                  name_bindings);
                    }
                }
            }
        }

        // We've successfully resolved the import. Write the results in.
        assert!(module_.import_resolutions.contains_key(&target));
        let import_resolution = module_.import_resolutions.get(&target);

        match value_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.value_target =
                    Some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!(~"value result should be known at this point");
            }
        }
        match type_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.type_target =
                    Some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail!(~"type result should be known at this point");
            }
        }

        let i = import_resolution;
        match (i.value_target, i.type_target) {
            // If this name wasn't found in either namespace, it's definitely
            // unresolved.
            (None, None) => { return Failed; }
            // If it's private, it's also unresolved.
            (Some(t), None) | (None, Some(t)) => {
                let bindings = &mut *t.bindings;
                match bindings.type_def {
                    Some(ref type_def) => {
                        if type_def.privacy == Private {
                            return Failed;
                        }
                    }
                    _ => ()
                }
                match bindings.value_def {
                    Some(ref value_def) => {
                        if value_def.privacy == Private {
                            return Failed;
                        }
                    }
                    _ => ()
                }
            }
            // It's also an error if there's both a type and a value with this
            // name, but both are private
            (Some(val), Some(ty)) => {
                match (val.bindings.value_def, ty.bindings.value_def) {
                    (Some(ref value_def), Some(ref type_def)) =>
                        if value_def.privacy == Private
                            && type_def.privacy == Private {
                            return Failed;
                        },
                    _ => ()
                }
            }
        }

        assert!(import_resolution.outstanding_references >= 1);
        import_resolution.outstanding_references -= 1;

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    // Resolves a glob import. Note that this function cannot fail; it either
    // succeeds or bails out (as importing * from an empty module or a module
    // that exports nothing is valid).
    fn resolve_glob_import(@mut self,
                           privacy: Privacy,
                           module_: @mut Module,
                           containing_module: @mut Module,
                           span: span)
                        -> ResolveResult<()> {
        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.
        debug!("(resolving glob import) resolving %? glob import", privacy);
        let state = @mut ImportState();

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).
        if !(*containing_module).all_imports_resolved() {
            debug!("(resolving glob import) target module has unresolved \
                    imports; bailing out");
            return Indeterminate;
        }

        assert!(containing_module.glob_count == 0);

        // Add all resolved imports from the containing module.
        for containing_module.import_resolutions.each
                |ident, target_import_resolution| {

            debug!("(resolving glob import) writing module resolution \
                    %? into `%s`",
                   target_import_resolution.type_target.is_none(),
                   self.module_to_str(module_));

            // Here we merge two import resolutions.
            match module_.import_resolutions.find(ident) {
                None if target_import_resolution.privacy == Public => {
                    // Simple: just copy the old import resolution.
                    let new_import_resolution =
                        @mut ImportResolution(privacy,
                                              target_import_resolution.span,
                                              state);
                    new_import_resolution.value_target =
                        copy target_import_resolution.value_target;
                    new_import_resolution.type_target =
                        copy target_import_resolution.type_target;

                    module_.import_resolutions.insert
                        (*ident, new_import_resolution);
                }
                None => { /* continue ... */ }
                Some(dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match target_import_resolution.value_target {
                        None => {
                            // Continue.
                        }
                        Some(copy value_target) => {
                            dest_import_resolution.value_target =
                                Some(value_target);
                        }
                    }
                    match target_import_resolution.type_target {
                        None => {
                            // Continue.
                        }
                        Some(copy type_target) => {
                            dest_import_resolution.type_target =
                                Some(type_target);
                        }
                    }
                }
            }
        }

        let merge_import_resolution = |ident,
                                       name_bindings: @mut NameBindings| {
            let dest_import_resolution;
            match module_.import_resolutions.find(ident) {
                None => {
                    // Create a new import resolution from this child.
                    dest_import_resolution = @mut ImportResolution(privacy,
                                                                   span,
                                                                   state);
                    module_.import_resolutions.insert
                        (*ident, dest_import_resolution);
                }
                Some(existing_import_resolution) => {
                    dest_import_resolution = *existing_import_resolution;
                }
            }

            debug!("(resolving glob import) writing resolution `%s` in `%s` \
                    to `%s`, privacy=%?",
                   *self.session.str_of(*ident),
                   self.module_to_str(containing_module),
                   self.module_to_str(module_),
                   copy dest_import_resolution.privacy);

            // Merge the child item into the import resolution.
            if name_bindings.defined_in_public_namespace(ValueNS) {
                debug!("(resolving glob import) ... for value target");
                dest_import_resolution.value_target =
                    Some(Target(containing_module, name_bindings));
            }
            if name_bindings.defined_in_public_namespace(TypeNS) {
                debug!("(resolving glob import) ... for type target");
                dest_import_resolution.type_target =
                    Some(Target(containing_module, name_bindings));
            }
        };

        // Add all children from the containing module.
        for containing_module.children.each |ident, name_bindings| {
            merge_import_resolution(ident, *name_bindings);
        }

        // Add external module children from the containing module.
        for containing_module.external_module_children.each
                |ident, module| {
            let name_bindings =
                @mut Resolver::create_name_bindings_from_module(*module);
            merge_import_resolution(ident, name_bindings);
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    /// Resolves the given module path from the given root `module_`.
    fn resolve_module_path_from_root(@mut self,
                                     module_: @mut Module,
                                     module_path: &[ident],
                                     index: uint,
                                     span: span,
                                     mut name_search_type: NameSearchType)
                                  -> ResolveResult<@mut Module> {
        let mut search_module = module_;
        let mut index = index;
        let module_path_len = module_path.len();

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
                    self.session.span_err(span, ~"unresolved name");
                    return Failed;
                }
                Indeterminate => {
                    debug!("(resolving module path for import) module \
                            resolution is indeterminate: %s",
                            *self.session.str_of(name));
                    return Indeterminate;
                }
                Success(target) => {
                    // Check to see whether there are type bindings, and, if
                    // so, whether there is a module within.
                    match target.bindings.type_def {
                        Some(copy type_def) => {
                            match type_def.module_def {
                                None => {
                                    // Not a module.
                                    self.session.span_err(span,
                                                          fmt!("not a \
                                                                module: %s",
                                                               *self.session.
                                                                   str_of(
                                                                    name)));
                                    return Failed;
                                }
                                Some(copy module_def) => {
                                    search_module = module_def;
                                }
                            }
                        }
                        None => {
                            // There are no type bindings at all.
                            self.session.span_err(span,
                                                  fmt!("not a module: %s",
                                                       *self.session.str_of(
                                                            name)));
                            return Failed;
                        }
                    }
                }
            }

            index += 1;

            // After the first element of the path, allow searching through
            // items and imports unconditionally. This allows things like:
            //
            // pub mod core {
            //     pub use vec;
            // }
            //
            // pub mod something_else {
            //     use core::vec;
            // }

            name_search_type = SearchItemsAndPublicImports;
        }

        return Success(search_module);
    }

    /// Attempts to resolve the module part of an import directive or path
    /// rooted at the given module.
    fn resolve_module_path_for_import(@mut self,
                                      module_: @mut Module,
                                      module_path: &[ident],
                                      use_lexical_scope: UseLexicalScopeFlag,
                                      span: span)
                                   -> ResolveResult<@mut Module> {
        let module_path_len = module_path.len();
        assert!(module_path_len > 0);

        debug!("(resolving module path for import) processing `%s` rooted at \
               `%s`",
               self.idents_to_str(module_path),
               self.module_to_str(module_));

        // Resolve the module prefix, if any.
        let module_prefix_result = self.resolve_module_prefix(module_,
                                                              module_path);

        let search_module;
        let start_index;
        match module_prefix_result {
            Failed => {
                self.session.span_err(span, ~"unresolved name");
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
                                self.session.span_err(span,
                                                      ~"unresolved name");
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
                            }
                        }
                    }
                }
            }
            Success(PrefixFound(containing_module, index)) => {
                search_module = containing_module;
                start_index = index;
            }
        }

        self.resolve_module_path_from_root(search_module,
                                           module_path,
                                           start_index,
                                           span,
                                           SearchItemsAndPublicImports)
    }

    /// Invariant: This must only be called during main resolution, not during
    /// import resolution.
    fn resolve_item_in_lexical_scope(@mut self,
                                     module_: @mut Module,
                                     name: ident,
                                     namespace: Namespace,
                                     search_through_modules:
                                        SearchThroughModulesFlag)
                                  -> ResolveResult<Target> {
        debug!("(resolving item in lexical scope) resolving `%s` in \
                namespace %? in `%s`",
               *self.session.str_of(name),
               namespace,
               self.module_to_str(module_));

        // The current module node is handled specially. First, check for
        // its immediate children.
        match module_.children.find(&name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                return Success(Target(module_, *name_bindings));
            }
            Some(_) | None => { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.
        match module_.import_resolutions.find(&name) {
            None => {
                // Not found; continue.
            }
            Some(import_resolution) => {
                match (*import_resolution).target_for_namespace(namespace) {
                    None => {
                        // Not found; continue.
                        debug!("(resolving item in lexical scope) found \
                                import resolution, but not in namespace %?",
                               namespace);
                    }
                    Some(target) => {
                        debug!("(resolving item in lexical scope) using \
                                import resolution");
                        import_resolution.state.used = true;
                        return Success(copy target);
                    }
                }
            }
        }

        // Search for external modules.
        if namespace == TypeNS {
            match module_.external_module_children.find(&name) {
                None => {}
                Some(module) => {
                    let name_bindings =
                        @mut Resolver::create_name_bindings_from_module(
                            *module);
                    return Success(Target(module_, name_bindings));
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
                            match search_module.kind {
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
                                              SearchItemsAndAllImports) {
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
                Success(target) => {
                    // We found the module.
                    return Success(copy target);
                }
            }
        }
    }

    /** Resolves a module name in the current lexical scope. */
    fn resolve_module_in_lexical_scope(@mut self,
                                       module_: @mut Module,
                                       name: ident)
                                    -> ResolveResult<@mut Module> {
        // If this module is an anonymous module, resolve the item in the
        // lexical scope. Otherwise, resolve the item from the crate root.
        let resolve_result = self.resolve_item_in_lexical_scope(
            module_, name, TypeNS, DontSearchThroughModules);
        match resolve_result {
            Success(target) => {
                let bindings = &mut *target.bindings;
                match bindings.type_def {
                    Some(ref type_def) => {
                        match (*type_def).module_def {
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

    /**
     * Returns the nearest normal module parent of the given module.
     */
    fn get_nearest_normal_module_parent(@mut self, module_: @mut Module)
                                     -> Option<@mut Module> {
        let mut module_ = module_;
        loop {
            match module_.parent_link {
                NoParentLink => return None,
                ModuleParentLink(new_module, _) |
                BlockParentLink(new_module, _) => {
                    match new_module.kind {
                        NormalModuleKind => return Some(new_module),
                        ExternModuleKind |
                        TraitModuleKind |
                        AnonymousModuleKind => module_ = new_module,
                    }
                }
            }
        }
    }

    /**
     * Returns the nearest normal module parent of the given module, or the
     * module itself if it is a normal module.
     */
    fn get_nearest_normal_module_parent_or_self(@mut self,
                                                module_: @mut Module)
                                             -> @mut Module {
        match module_.kind {
            NormalModuleKind => return module_,
            ExternModuleKind | TraitModuleKind | AnonymousModuleKind => {
                match self.get_nearest_normal_module_parent(module_) {
                    None => module_,
                    Some(new_module) => new_module
                }
            }
        }
    }

    /**
     * Resolves a "module prefix". A module prefix is one of (a) `self::`;
     * (b) some chain of `super::`.
     */
    fn resolve_module_prefix(@mut self,
                             module_: @mut Module,
                             module_path: &[ident])
                          -> ResolveResult<ModulePrefixResult> {
        let interner = self.session.parse_sess.interner;

        // Start at the current module if we see `self` or `super`, or at the
        // top of the crate otherwise.
        let mut containing_module;
        let mut i;
        if *interner.get(module_path[0]) == ~"self" {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 1;
        } else if *interner.get(module_path[0]) == ~"super" {
            containing_module =
                self.get_nearest_normal_module_parent_or_self(module_);
            i = 0;  // We'll handle `super` below.
        } else {
            return Success(NoPrefixFound);
        }

        // Now loop through all the `super`s we find.
        while i < module_path.len() &&
                *interner.get(module_path[i]) == ~"super" {
            debug!("(resolving module prefix) resolving `super` at %s",
                   self.module_to_str(containing_module));
            match self.get_nearest_normal_module_parent(containing_module) {
                None => return Failed,
                Some(new_module) => {
                    containing_module = new_module;
                    i += 1;
                }
            }
        }

        debug!("(resolving module prefix) finished resolving prefix at %s",
               self.module_to_str(containing_module));

        return Success(PrefixFound(containing_module, i));
    }

    /// Attempts to resolve the supplied name in the given module for the
    /// given namespace. If successful, returns the target corresponding to
    /// the name.
    fn resolve_name_in_module(@mut self,
                              module_: @mut Module,
                              name: ident,
                              namespace: Namespace,
                              name_search_type: NameSearchType)
                           -> ResolveResult<Target> {
        debug!("(resolving name in module) resolving `%s` in `%s`",
               *self.session.str_of(name),
               self.module_to_str(module_));

        // First, check the direct children of the module.
        match module_.children.find(&name) {
            Some(name_bindings)
                    if name_bindings.defined_in_namespace(namespace) => {
                debug!("(resolving name in module) found node as child");
                return Success(Target(module_, *name_bindings));
            }
            Some(_) | None => {
                // Continue.
            }
        }

        // Next, check the module's imports if necessary.

        // If this is a search of all imports, we should be done with glob
        // resolution at this point.
        if name_search_type == SearchItemsAndAllImports {
            assert!(module_.glob_count == 0);
        }

        // Check the list of resolved imports.
        match module_.import_resolutions.find(&name) {
            Some(import_resolution) => {
                if import_resolution.privacy == Public &&
                        import_resolution.outstanding_references != 0 {
                    debug!("(resolving name in module) import \
                            unresolved; bailing out");
                    return Indeterminate;
                }

                match import_resolution.target_for_namespace(namespace) {
                    None => {
                        debug!("(resolving name in module) name found, \
                                but not in namespace %?",
                               namespace);
                    }
                    Some(target)
                            if name_search_type ==
                                SearchItemsAndAllImports ||
                            import_resolution.privacy == Public => {
                        debug!("(resolving name in module) resolved to \
                                import");
                        import_resolution.state.used = true;
                        return Success(copy target);
                    }
                    Some(_) => {
                        debug!("(resolving name in module) name found, \
                                but not public");
                    }
                }
            }
            None => {} // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            match module_.external_module_children.find(&name) {
                None => {}
                Some(module) => {
                    let name_bindings =
                        @mut Resolver::create_name_bindings_from_module(
                            *module);
                    return Success(Target(module_, name_bindings));
                }
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve %s",
               *self.session.str_of(name));
        return Failed;
    }

    fn report_unresolved_imports(@mut self, module_: @mut Module) {
        let index = module_.resolved_import_count;
        let imports: &mut ~[@ImportDirective] = &mut *module_.imports;
        let import_count = imports.len();
        if index != import_count {
            let sn = self.session.codemap.span_to_snippet(imports[index].span);
            if str::contains(sn, "::") {
                self.session.span_err(imports[index].span, ~"unresolved import");
            } else {
                let err = fmt!("unresolved import (maybe you meant `%s::*`?)",
                               sn.slice(0, sn.len() - 1)); // -1 to adjust for semicolon
                self.session.span_err(imports[index].span, err);
            }
        }

        // Descend into children and anonymous children.
        for module_.children.each_value |&child_node| {
            match child_node.get_module_if_available() {
                None => {
                    // Continue.
                }
                Some(child_module) => {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for module_.anonymous_children.each_value |&module_| {
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

    fn record_exports(@mut self) {
        let root_module = self.graph_root.get_module();
        self.record_exports_for_module_subtree(root_module);
    }

    fn record_exports_for_module_subtree(@mut self, module_: @mut Module) {
        // If this isn't a local crate, then bail out. We don't need to record
        // exports for nonlocal crates.

        match module_.def_id {
            Some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
                debug!("(recording exports for module subtree) recording \
                        exports for local module");
            }
            None => {
                // Record exports for the root module.
                debug!("(recording exports for module subtree) recording \
                        exports for root module");
            }
            Some(_) => {
                // Bail out.
                debug!("(recording exports for module subtree) not recording \
                        exports for `%s`",
                       self.module_to_str(module_));
                return;
            }
        }

        self.record_exports_for_module(module_);

        for module_.children.each_value |&child_name_bindings| {
            match child_name_bindings.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.record_exports_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each_value |&child_module| {
            self.record_exports_for_module_subtree(child_module);
        }
    }

    fn record_exports_for_module(@mut self, module_: @mut Module) {
        let mut exports2 = ~[];

        self.add_exports_for_module(&mut exports2, module_);
        match /*bad*/copy module_.def_id {
            Some(def_id) => {
                self.export_map2.insert(def_id.node, exports2);
                debug!("(computing exports) writing exports for %d (some)",
                       def_id.node);
            }
            None => {}
        }
    }

    fn add_exports_of_namebindings(@mut self,
                                   exports2: &mut ~[Export2],
                                   ident: ident,
                                   namebindings: @mut NameBindings,
                                   ns: Namespace,
                                   reexport: bool) {
        match (namebindings.def_for_namespace(ns),
               namebindings.privacy_for_namespace(ns)) {
            (Some(d), Some(Public)) => {
                debug!("(computing exports) YES: %s '%s' => %?",
                       if reexport { ~"reexport" } else { ~"export"},
                       *self.session.str_of(ident),
                       def_id_of_def(d));
                exports2.push(Export2 {
                    reexport: reexport,
                    name: self.session.str_of(ident),
                    def_id: def_id_of_def(d)
                });
            }
            (Some(_), Some(privacy)) => {
                debug!("(computing reexports) NO: privacy %?", privacy);
            }
            (d_opt, p_opt) => {
                debug!("(computing reexports) NO: %?, %?", d_opt, p_opt);
            }
        }
    }

    fn add_exports_for_module(@mut self,
                              exports2: &mut ~[Export2],
                              module_: @mut Module) {
        for module_.children.each |ident, namebindings| {
            debug!("(computing exports) maybe export '%s'",
                   *self.session.str_of(*ident));
            self.add_exports_of_namebindings(&mut *exports2,
                                             *ident,
                                             *namebindings,
                                             TypeNS,
                                             false);
            self.add_exports_of_namebindings(&mut *exports2,
                                             *ident,
                                             *namebindings,
                                             ValueNS,
                                             false);
        }

        for module_.import_resolutions.each |ident, importresolution| {
            if importresolution.privacy != Public {
                debug!("(computing exports) not reexporting private `%s`",
                       *self.session.str_of(*ident));
                loop;
            }
            for [ TypeNS, ValueNS ].each |ns| {
                match importresolution.target_for_namespace(*ns) {
                    Some(target) => {
                        debug!("(computing exports) maybe reexport '%s'",
                               *self.session.str_of(*ident));
                        self.add_exports_of_namebindings(&mut *exports2,
                                                         *ident,
                                                         target.bindings,
                                                         *ns,
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

    fn with_scope(@mut self, name: Option<ident>, f: &fn()) {
        let orig_module = self.current_module;

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                match orig_module.children.find(&name) {
                    None => {
                        debug!("!!! (with scope) didn't find `%s` in `%s`",
                               *self.session.str_of(name),
                               self.module_to_str(orig_module));
                    }
                    Some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            None => {
                                debug!("!!! (with scope) didn't find module \
                                        for `%s` in `%s`",
                                       *self.session.str_of(name),
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

        f();

        self.current_module = orig_module;
    }

    // Wraps the given definition in the appropriate number of `def_upvar`
    // wrappers.

    fn upvarify(@mut self,
                ribs: &mut ~[@Rib],
                rib_index: uint,
                def_like: def_like,
                span: span,
                allow_capturing_self: AllowCapturingSelfFlag)
             -> Option<def_like> {
        let mut def;
        let is_ty_param;

        match def_like {
            dl_def(d @ def_local(*)) | dl_def(d @ def_upvar(*)) |
            dl_def(d @ def_arg(*)) | dl_def(d @ def_binding(*)) => {
                def = d;
                is_ty_param = false;
            }
            dl_def(d @ def_ty_param(*)) => {
                def = d;
                is_ty_param = true;
            }
            dl_def(d @ def_self(*))
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
                        def = def_upvar(def_id_of_def(def).node,
                                        @def,
                                        function_id,
                                        body_id);
                    }
                }
                MethodRibKind(item_id, _) => {
                  // If the def is a ty param, and came from the parent
                  // item, it's ok
                  match def {
                    def_ty_param(did, _)
                        if self.def_map.find(&did.node).map_consume(|x| *x)
                            == Some(def_typaram_binder(item_id)) => {
                      // ok
                    }
                    _ => {
                    if !is_ty_param {
                        // This was an attempt to access an upvar inside a
                        // named function item. This is not allowed, so we
                        // report an error.

                        self.session.span_err(
                            span,
                            ~"attempted dynamic environment-capture");
                    } else {
                        // This was an attempt to use a type parameter outside
                        // its scope.

                        self.session.span_err(span,
                                              ~"attempt to use a type \
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

                        self.session.span_err(
                            span,
                            ~"attempted dynamic environment-capture");
                    } else {
                        // This was an attempt to use a type parameter outside
                        // its scope.

                        self.session.span_err(span,
                                              ~"attempt to use a type \
                                               argument out of scope");
                    }

                    return None;
                }
                ConstantItemRibKind => {
                    // Still doesn't deal with upvars
                    self.session.span_err(span,
                                          ~"attempt to use a non-constant \
                                            value in a constant");

                }
            }

            rib_index += 1;
        }

        return Some(dl_def(def));
    }

    fn search_ribs(@mut self,
                   ribs: &mut ~[@Rib],
                   name: ident,
                   span: span,
                   allow_capturing_self: AllowCapturingSelfFlag)
                -> Option<def_like> {
        // FIXME #4950: This should not use a while loop.
        // FIXME #4950: Try caching?

        let mut i = ribs.len();
        while i != 0 {
            i -= 1;
            match ribs[i].bindings.find(&name) {
                Some(&def_like) => {
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

    fn resolve_crate(@mut self) {
        debug!("(resolving crate) starting");

        visit_crate(self.crate, (), mk_vt(@Visitor {
            visit_item: |item, _context, visitor|
                self.resolve_item(item, visitor),
            visit_arm: |arm, _context, visitor|
                self.resolve_arm(arm, visitor),
            visit_block: |block, _context, visitor|
                self.resolve_block(block, visitor),
            visit_expr: |expr, _context, visitor|
                self.resolve_expr(expr, visitor),
            visit_local: |local, _context, visitor|
                self.resolve_local(local, visitor),
            visit_ty: |ty, _context, visitor|
                self.resolve_type(ty, visitor),
            .. *default_visitor()
        }));
    }

    fn resolve_item(@mut self, item: @item, visitor: ResolveVisitor) {
        debug!("(resolving item) resolving %s",
               *self.session.str_of(item.ident));

        // Items with the !resolve_unexported attribute are X-ray contexts.
        // This is used to allow the test runner to run unexported tests.
        let orig_xray_flag = self.xray_context;
        if contains_name(attr_metas(item.attrs),
                         ~"!resolve_unexported") {
            self.xray_context = Xray;
        }

        match item.node {

            // enum item: resolve all the variants' discrs,
            // then resolve the ty params
            item_enum(ref enum_def, ref generics) => {
                for (*enum_def).variants.each() |variant| {
                    for variant.node.disr_expr.each |dis_expr| {
                        // resolve the discriminator expr
                        // as a constant
                        self.with_constant_rib(|| {
                            self.resolve_expr(*dis_expr, visitor);
                        });
                    }
                }

                // n.b. the discr expr gets visted twice.
                // but maybe it's okay since the first time will signal an
                // error if there is one? -- tjc
                do self.with_type_parameter_rib(
                    HasTypeParameters(
                        generics, item.id, 0, NormalRibKind)) {
                    visit_item(item, (), visitor);
                }
            }

            item_ty(_, ref generics) => {
                do self.with_type_parameter_rib
                        (HasTypeParameters(generics, item.id, 0,
                                           NormalRibKind))
                        || {

                    visit_item(item, (), visitor);
                }
            }

            item_impl(ref generics,
                      implemented_traits,
                      self_type,
                      ref methods) => {
                self.resolve_implementation(item.id,
                                            generics,
                                            implemented_traits,
                                            self_type,
                                            *methods,
                                            visitor);
            }

            item_trait(ref generics, ref traits, ref methods) => {
                // Create a new rib for the self type.
                let self_type_rib = @Rib(NormalRibKind);
                self.type_ribs.push(self_type_rib);
                self_type_rib.bindings.insert(self.type_self_ident,
                                              dl_def(def_self_ty(item.id)));

                // Create a new rib for the trait-wide type parameters.
                do self.with_type_parameter_rib
                        (HasTypeParameters(generics, item.id, 0,
                                           NormalRibKind)) {

                    self.resolve_type_parameters(&generics.ty_params,
                                                 visitor);

                    // Resolve derived traits.
                    for traits.each |trt| {
                        match self.resolve_path(trt.path, TypeNS, true,
                                                visitor) {
                            None =>
                                self.session.span_err(trt.path.span,
                                                      ~"attempt to derive a \
                                                       nonexistent trait"),
                            Some(def) => {
                                // Write a mapping from the trait ID to the
                                // definition of the trait into the definition
                                // map.

                                debug!("(resolving trait) found trait def: \
                                       %?", def);

                                self.record_def(trt.ref_id, def);
                            }
                        }
                    }

                    for (*methods).each |method| {
                        // Create a new rib for the method-specific type
                        // parameters.
                        //
                        // FIXME #4951: Do we need a node ID here?

                        match *method {
                          required(ref ty_m) => {
                            do self.with_type_parameter_rib
                                (HasTypeParameters(&ty_m.generics,
                                                   item.id,
                                                   generics.ty_params.len(),
                                        MethodRibKind(item.id, Required))) {

                                // Resolve the method-specific type
                                // parameters.
                                self.resolve_type_parameters(
                                    &ty_m.generics.ty_params,
                                    visitor);

                                for ty_m.decl.inputs.each |argument| {
                                    self.resolve_type(argument.ty, visitor);
                                }

                                self.resolve_type(ty_m.decl.output, visitor);
                            }
                          }
                          provided(m) => {
                              self.resolve_method(MethodRibKind(item.id,
                                                     Provided(m.id)),
                                                  m,
                                                  generics.ty_params.len(),
                                                  visitor)
                          }
                        }
                    }
                }

                self.type_ribs.pop();
            }

            item_struct(ref struct_def, ref generics) => {
                self.resolve_struct(item.id,
                                    generics,
                                    struct_def.fields,
                                    &struct_def.dtor,
                                    visitor);
            }

            item_mod(ref module_) => {
                do self.with_scope(Some(item.ident)) {
                    self.resolve_module(module_, item.span, item.ident,
                                        item.id, visitor);
                }
            }

            item_foreign_mod(ref foreign_module) => {
                do self.with_scope(Some(item.ident)) {
                    for foreign_module.items.each |foreign_item| {
                        match foreign_item.node {
                            foreign_item_fn(_, _, ref generics) => {
                                self.with_type_parameter_rib(
                                    HasTypeParameters(
                                        generics, foreign_item.id, 0,
                                        NormalRibKind),
                                    || visit_foreign_item(*foreign_item, (),
                                                          visitor));
                            }
                            foreign_item_const(_) => {
                                visit_foreign_item(*foreign_item, (),
                                                   visitor);
                            }
                        }
                    }
                }
            }

            item_fn(ref fn_decl, _, _, ref generics, ref block) => {
                // If this is the main function, we must record it in the
                // session.

                // FIXME #4404 android JNI hacks
                if !*self.session.building_library ||
                    self.session.targ_cfg.os == session::os_android {

                    if self.attr_main_fn.is_none() &&
                           item.ident == special_idents::main {

                        self.main_fns.push(Some((item.id, item.span)));
                    }

                    if attrs_contains_name(item.attrs, ~"main") {
                        if self.attr_main_fn.is_none() {
                            self.attr_main_fn = Some((item.id, item.span));
                        } else {
                            self.session.span_err(
                                    item.span,
                                    ~"multiple 'main' functions");
                        }
                    }

                    if attrs_contains_name(item.attrs, ~"start") {
                        if self.start_fn.is_none() {
                            self.start_fn = Some((item.id, item.span));
                        } else {
                            self.session.span_err(
                                    item.span,
                                    ~"multiple 'start' functions");
                        }
                    }
                }

                self.resolve_function(OpaqueFunctionRibKind,
                                      Some(fn_decl),
                                      HasTypeParameters
                                        (generics,
                                         item.id,
                                         0,
                                         OpaqueFunctionRibKind),
                                      block,
                                      NoSelfBinding,
                                      visitor);
            }

            item_const(*) => {
                self.with_constant_rib(|| {
                    visit_item(item, (), visitor);
                });
            }

          item_mac(*) => {
            fail!(~"item macros unimplemented")
          }
        }

        self.xray_context = orig_xray_flag;
    }

    fn with_type_parameter_rib(@mut self,
                               type_parameters: TypeParameters,
                               f: &fn()) {
        match type_parameters {
            HasTypeParameters(generics, node_id, initial_index,
                              rib_kind) => {

                let function_type_rib = @Rib(rib_kind);
                self.type_ribs.push(function_type_rib);

                for generics.ty_params.eachi |index, type_parameter| {
                    let name = type_parameter.ident;
                    debug!("with_type_parameter_rib: %d %d", node_id,
                           type_parameter.id);
                    let def_like = dl_def(def_ty_param
                        (local_def(type_parameter.id),
                         index + initial_index));
                    // Associate this type parameter with
                    // the item that bound it
                    self.record_def(type_parameter.id,
                                    def_typaram_binder(node_id));
                    function_type_rib.bindings.insert(name, def_like);
                }
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f();

        match type_parameters {
            HasTypeParameters(*) => {
                self.type_ribs.pop();
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }
    }

    fn with_label_rib(@mut self, f: &fn()) {
        self.label_ribs.push(@Rib(NormalRibKind));
        f();
        self.label_ribs.pop();
    }

    fn with_constant_rib(@mut self, f: &fn()) {
        self.value_ribs.push(@Rib(ConstantItemRibKind));
        f();
        self.value_ribs.pop();
    }

    fn resolve_function(@mut self,
                        rib_kind: RibKind,
                        optional_declaration: Option<&fn_decl>,
                        type_parameters: TypeParameters,
                        block: &blk,
                        self_binding: SelfBinding,
                        visitor: ResolveVisitor) {
        // Create a value rib for the function.
        let function_value_rib = @Rib(rib_kind);
        self.value_ribs.push(function_value_rib);

        // Create a label rib for the function.
        let function_label_rib = @Rib(rib_kind);
        self.label_ribs.push(function_label_rib);

        // If this function has type parameters, add them now.
        do self.with_type_parameter_rib(type_parameters) {
            // Resolve the type parameters.
            match type_parameters {
                NoTypeParameters => {
                    // Continue.
                }
                HasTypeParameters(ref generics, _, _, _) => {
                    self.resolve_type_parameters(&generics.ty_params,
                                                 visitor);
                }
            }

            // Add self to the rib, if necessary.
            match self_binding {
                NoSelfBinding => {
                    // Nothing to do.
                }
                HasSelfBinding(self_node_id, is_implicit) => {
                    let def_like = dl_def(def_self(self_node_id,
                                                   is_implicit));
                    (*function_value_rib).bindings.insert(self.self_ident,
                                                          def_like);
                }
            }

            // Add each argument to the rib.
            match optional_declaration {
                None => {
                    // Nothing to do.
                }
                Some(declaration) => {
                    for declaration.inputs.each |argument| {
                        let binding_mode = ArgumentIrrefutableMode;
                        let mutability =
                            if argument.is_mutbl {Mutable} else {Immutable};
                        self.resolve_pattern(argument.pat,
                                             binding_mode,
                                             mutability,
                                             None,
                                             visitor);

                        self.resolve_type(argument.ty, visitor);

                        debug!("(resolving function) recorded argument");
                    }

                    self.resolve_type(declaration.output, visitor);
                }
            }

            // Resolve the function body.
            self.resolve_block(block, visitor);

            debug!("(resolving function) leaving function");
        }

        self.label_ribs.pop();
        self.value_ribs.pop();
    }

    fn resolve_type_parameters(@mut self,
                               type_parameters: &OptVec<TyParam>,
                               visitor: ResolveVisitor) {
        for type_parameters.each |type_parameter| {
            for type_parameter.bounds.each |&bound| {
                match bound {
                    TraitTyParamBound(tref) => {
                        self.resolve_trait_reference(tref, visitor)
                    }
                    RegionTyParamBound => {}
                }
            }
        }
    }

    fn resolve_trait_reference(@mut self,
                               trait_reference: &trait_ref,
                               visitor: ResolveVisitor) {
        match self.resolve_path(trait_reference.path, TypeNS, true, visitor) {
            None => {
                self.session.span_err(trait_reference.path.span,
                                      ~"attempt to implement an \
                                        unknown trait");
            }
            Some(def) => {
                self.record_def(trait_reference.ref_id, def);
            }
        }
    }

    fn resolve_struct(@mut self,
                      id: node_id,
                      generics: &Generics,
                      fields: &[@struct_field],
                      optional_destructor: &Option<struct_dtor>,
                      visitor: ResolveVisitor) {
        // If applicable, create a rib for the type parameters.
        do self.with_type_parameter_rib(HasTypeParameters
                                        (generics, id, 0,
                                         OpaqueFunctionRibKind)) {

            // Resolve the type parameters.
            self.resolve_type_parameters(&generics.ty_params, visitor);

            // Resolve fields.
            for fields.each |field| {
                self.resolve_type(field.node.ty, visitor);
            }

            // Resolve the destructor, if applicable.
            match *optional_destructor {
                None => {
                    // Nothing to do.
                }
                Some(ref destructor) => {
                    self.resolve_function(NormalRibKind,
                                          None,
                                          NoTypeParameters,
                                          &destructor.node.body,
                                          HasSelfBinding
                                            ((*destructor).node.self_id,
                                             true),
                                          visitor);
                }
            }
        }
    }

    // Does this really need to take a RibKind or is it always going
    // to be NormalRibKind?
    fn resolve_method(@mut self,
                      rib_kind: RibKind,
                      method: @method,
                      outer_type_parameter_count: uint,
                      visitor: ResolveVisitor) {
        let method_generics = &method.generics;
        let type_parameters =
            HasTypeParameters(method_generics,
                              method.id,
                              outer_type_parameter_count,
                              rib_kind);
        // we only have self ty if it is a non static method
        let self_binding = match method.self_ty.node {
          sty_static => { NoSelfBinding }
          _ => { HasSelfBinding(method.self_id, false) }
        };

        self.resolve_function(rib_kind,
                              Some(&method.decl),
                              type_parameters,
                              &method.body,
                              self_binding,
                              visitor);
    }

    fn resolve_implementation(@mut self,
                              id: node_id,
                              generics: &Generics,
                              opt_trait_reference: Option<@trait_ref>,
                              self_type: @Ty,
                              methods: &[@method],
                              visitor: ResolveVisitor) {
        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = generics.ty_params.len();
        do self.with_type_parameter_rib(HasTypeParameters
                                        (generics, id, 0,
                                         NormalRibKind)) {
            // Resolve the type parameters.
            self.resolve_type_parameters(&generics.ty_params,
                                         visitor);

            // Resolve the trait reference, if necessary.
            let original_trait_refs;
            match opt_trait_reference {
                Some(trait_reference) => {
                    self.resolve_trait_reference(trait_reference, visitor);

                    // Record the current set of trait references.
                    let mut new_trait_refs = ~[];
                    for self.def_map.find(&trait_reference.ref_id).each |&def| {
                        new_trait_refs.push(def_id_of_def(*def));
                    }
                    original_trait_refs = Some(util::replace(
                        &mut self.current_trait_refs,
                        Some(new_trait_refs)));
                }
                None => {
                    original_trait_refs = None;
                }
            }

            // Resolve the self type.
            self.resolve_type(self_type, visitor);

            for methods.each |method| {
                // We also need a new scope for the method-specific
                // type parameters.
                self.resolve_method(MethodRibKind(
                    id,
                    Provided(method.id)),
                    *method,
                    outer_type_parameter_count,
                    visitor);
/*
                    let borrowed_type_parameters = &method.tps;
                    self.resolve_function(MethodRibKind(
                                          id,
                                          Provided(method.id)),
                                          Some(@method.decl),
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
                Some(r) => { self.current_trait_refs = r; }
                None => ()
            }
        }
    }

    fn resolve_module(@mut self,
                      module_: &_mod,
                      span: span,
                      _name: ident,
                      id: node_id,
                      visitor: ResolveVisitor) {
        // Write the implementations in scope into the module metadata.
        debug!("(resolving module) resolving module ID %d", id);
        visit_mod(module_, span, id, (), visitor);
    }

    fn resolve_local(@mut self, local: @local, visitor: ResolveVisitor) {
        let mutability = if local.node.is_mutbl {Mutable} else {Immutable};

        // Resolve the type.
        self.resolve_type(local.node.ty, visitor);

        // Resolve the initializer, if necessary.
        match local.node.init {
            None => {
                // Nothing to do.
            }
            Some(initializer) => {
                self.resolve_expr(initializer, visitor);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern(local.node.pat, LocalIrrefutableMode, mutability,
                             None, visitor);
    }

    fn binding_mode_map(@mut self, pat: @pat) -> BindingMap {
        let mut result = HashMap::new();
        do pat_bindings(self.def_map, pat) |binding_mode, _id, sp, path| {
            let ident = path_to_ident(path);
            result.insert(ident,
                          binding_info {span: sp,
                                        binding_mode: binding_mode});
        }
        return result;
    }

    fn check_consistent_bindings(@mut self, arm: &arm) {
        if arm.pats.len() == 0 { return; }
        let map_0 = self.binding_mode_map(arm.pats[0]);
        for arm.pats.eachi() |i, p| {
            let map_i = self.binding_mode_map(*p);

            for map_0.each |&key, &binding_0| {
                match map_i.find(&key) {
                  None => {
                    self.session.span_err(
                        p.span,
                        fmt!("variable `%s` from pattern #1 is \
                                  not bound in pattern #%u",
                             *self.session.str_of(key), i + 1));
                  }
                  Some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.session.span_err(
                            binding_i.span,
                            fmt!("variable `%s` is bound with different \
                                      mode in pattern #%u than in pattern #1",
                                 *self.session.str_of(key), i + 1));
                    }
                  }
                }
            }

            for map_i.each |&key, &binding| {
                if !map_0.contains_key(&key) {
                    self.session.span_err(
                        binding.span,
                        fmt!("variable `%s` from pattern #%u is \
                                  not bound in pattern #1",
                             *self.session.str_of(key), i + 1));
                }
            }
        }
    }

    fn resolve_arm(@mut self, arm: &arm, visitor: ResolveVisitor) {
        self.value_ribs.push(@Rib(NormalRibKind));

        let bindings_list = @mut HashMap::new();
        for arm.pats.each |pattern| {
            self.resolve_pattern(*pattern, RefutableMode, Immutable,
                                 Some(bindings_list), visitor);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        visit_expr_opt(arm.guard, (), visitor);
        self.resolve_block(&arm.body, visitor);

        self.value_ribs.pop();
    }

    fn resolve_block(@mut self, block: &blk, visitor: ResolveVisitor) {
        debug!("(resolving block) entering block");
        self.value_ribs.push(@Rib(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        match self.current_module.anonymous_children.find(&block.node.id) {
            None => { /* Nothing to do. */ }
            Some(&anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module;
            }
        }

        // Descend into the block.
        visit_block(block, (), visitor);

        // Move back up.
        self.current_module = orig_module;

        self.value_ribs.pop();
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(@mut self, ty: @Ty, visitor: ResolveVisitor) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            ty_path(path, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // scopes looking for it.
                let mut result_def = None;

                // First, check to see whether the name is a primitive type.
                if path.idents.len() == 1 {
                    let name = *path.idents.last();

                    match self.primitive_type_table
                            .primitive_types
                            .find(&name) {

                        Some(&primitive_type) => {
                            result_def =
                                Some(def_prim_ty(primitive_type));
                        }
                        None => {
                            // Continue.
                        }
                    }
                }

                match result_def {
                    None => {
                        match self.resolve_path(path, TypeNS, true, visitor) {
                            Some(def) => {
                                debug!("(resolving type) resolved `%s` to \
                                        type %?",
                                       *self.session.str_of(
                                            *path.idents.last()),
                                       def);
                                result_def = Some(def);
                            }
                            None => {
                                result_def = None;
                            }
                        }
                    }
                    Some(_) => {
                        // Continue.
                    }
                }

                match result_def {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving type) writing resolution for `%s` \
                                (id %d)",
                               self.idents_to_str(path.idents),
                               path_id);
                        self.record_def(path_id, def);
                    }
                    None => {
                        self.session.span_err
                            (ty.span, fmt!("use of undeclared type name `%s`",
                                           self.idents_to_str(path.idents)));
                    }
                }
            }

            _ => {
                // Just resolve embedded types.
                visit_ty(ty, (), visitor);
            }
        }
    }

    fn resolve_pattern(@mut self,
                       pattern: @pat,
                       mode: PatternBindingMode,
                       mutability: Mutability,
                       // Maps idents to the node ID for the (outermost)
                       // pattern that binds them
                       bindings_list: Option<@mut HashMap<ident,node_id>>,
                       visitor: ResolveVisitor) {
        let pat_id = pattern.id;
        do walk_pat(pattern) |pattern| {
            match pattern.node {
                pat_ident(binding_mode, path, _)
                        if !path.global && path.idents.len() == 1 => {

                    // The meaning of pat_ident with no type parameters
                    // depends on whether an enum variant or unit-like struct
                    // with that name is in scope. The probing lookup has to
                    // be careful not to emit spurious errors. Only matching
                    // patterns (match) can match nullary variants or
                    // unit-like structs. For binding patterns (let), matching
                    // such a value is simply disallowed (since it's rarely
                    // what you want).

                    let ident = path.idents[0];

                    match self.resolve_bare_identifier_pattern(ident) {
                        FoundStructOrEnumVariant(def)
                                if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `%s` to \
                                    struct or enum variant",
                                    *self.session.str_of(ident));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "an enum variant");
                            self.record_def(pattern.id, def);
                        }
                        FoundStructOrEnumVariant(_) => {
                            self.session.span_err(pattern.span,
                                                  fmt!("declaration of `%s` \
                                                        shadows an enum \
                                                        variant or unit-like \
                                                        struct in scope",
                                                        *self.session
                                                            .str_of(ident)));
                        }
                        FoundConst(def) if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `%s` to \
                                    constant",
                                    *self.session.str_of(ident));

                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "a constant");
                            self.record_def(pattern.id, def);
                        }
                        FoundConst(_) => {
                            self.session.span_err(pattern.span,
                                                  ~"only refutable patterns \
                                                    allowed here");
                        }
                        BareIdentifierPatternUnresolved => {
                            debug!("(resolving pattern) binding `%s`",
                                   *self.session.str_of(ident));

                            let is_mutable = mutability == Mutable;

                            let def = match mode {
                                RefutableMode => {
                                    // For pattern arms, we must use
                                    // `def_binding` definitions.

                                    def_binding(pattern.id, binding_mode)
                                }
                                LocalIrrefutableMode => {
                                    // But for locals, we use `def_local`.
                                    def_local(pattern.id, is_mutable)
                                }
                                ArgumentIrrefutableMode => {
                                    // And for function arguments, `def_arg`.
                                    def_arg(pattern.id, is_mutable)
                                }
                            };

                            // Record the definition so that later passes
                            // will be able to distinguish variants from
                            // locals in patterns.

                            self.record_def(pattern.id, def);

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)

                            match bindings_list {
                                Some(bindings_list)
                                if !bindings_list.contains_key(&ident) => {
                                    let this = &mut *self;
                                    let last_rib = this.value_ribs[
                                            this.value_ribs.len() - 1];
                                    last_rib.bindings.insert(ident,
                                                             dl_def(def));
                                    bindings_list.insert(ident, pat_id);
                                }
                                Some(b) => {
                                  if b.find(&ident) == Some(&pat_id) {
                                      // Then this is a duplicate variable
                                      // in the same disjunct, which is an
                                      // error
                                     self.session.span_err(pattern.span,
                                       fmt!("Identifier %s is bound more \
                                             than once in the same pattern",
                                            path_to_str(path, self.session
                                                        .intr())));
                                  }
                                  // Not bound in the same pattern: do nothing
                                }
                                None => {
                                    let this = &mut *self;
                                    let last_rib = this.value_ribs[
                                            this.value_ribs.len() - 1];
                                    last_rib.bindings.insert(ident,
                                                             dl_def(def));
                                }
                            }
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(*ty, visitor);
                    }
                }

                pat_ident(binding_mode, path, _) => {
                    // This must be an enum variant, struct, or constant.
                    match self.resolve_path(path, ValueNS, false, visitor) {
                        Some(def @ def_variant(*)) |
                                Some(def @ def_struct(*)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some(def @ def_const(*)) => {
                            self.enforce_default_binding_mode(
                                pattern,
                                binding_mode,
                                "a constant");
                            self.record_def(pattern.id, def);
                        }
                        Some(_) => {
                            self.session.span_err(
                                path.span,
                                fmt!("not an enum variant or constant: %s",
                                     *self.session.str_of(
                                         *path.idents.last())));
                        }
                        None => {
                            self.session.span_err(path.span,
                                                  ~"unresolved enum variant");
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(*ty, visitor);
                    }
                }

                pat_enum(path, _) => {
                    // This must be an enum variant, struct or const.
                    match self.resolve_path(path, ValueNS, false, visitor) {
                        Some(def @ def_fn(*))      |
                        Some(def @ def_variant(*)) |
                        Some(def @ def_struct(*))  |
                        Some(def @ def_const(*)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some(_) => {
                            self.session.span_err(
                                path.span,
                                fmt!("not an enum variant, struct or const: %s",
                                     *self.session.str_of(
                                         *path.idents.last())));
                        }
                        None => {
                            self.session.span_err(path.span,
                                                  ~"unresolved enum variant, \
                                                    struct or const");
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(*ty, visitor);
                    }
                }

                pat_lit(expr) => {
                    self.resolve_expr(expr, visitor);
                }

                pat_range(first_expr, last_expr) => {
                    self.resolve_expr(first_expr, visitor);
                    self.resolve_expr(last_expr, visitor);
                }

                pat_struct(path, _, _) => {
                    let structs: &mut HashSet<def_id> = &mut self.structs;
                    match self.resolve_path(path, TypeNS, false, visitor) {
                        Some(def_ty(class_id))
                                if structs.contains(&class_id) => {
                            let class_def = def_struct(class_id);
                            self.record_def(pattern.id, class_def);
                        }
                        Some(definition @ def_struct(class_id))
                                if structs.contains(&class_id) => {
                            self.record_def(pattern.id, definition);
                        }
                        Some(definition @ def_variant(_, variant_id))
                                if structs.contains(&variant_id) => {
                            self.record_def(pattern.id, definition);
                        }
                        result => {
                            debug!("(resolving pattern) didn't find struct \
                                    def: %?", result);
                            self.session.span_err(
                                path.span,
                                fmt!("`%s` does not name a structure",
                                     self.idents_to_str(path.idents)));
                        }
                    }
                }

                _ => {
                    // Nothing to do.
                }
            }
        }
    }

    fn resolve_bare_identifier_pattern(@mut self, name: ident)
                                    -> BareIdentifierPatternResolution {
        match self.resolve_item_in_lexical_scope(self.current_module,
                                                 name,
                                                 ValueNS,
                                                 SearchThroughModules) {
            Success(target) => {
                match target.bindings.value_def {
                    None => {
                        fail!(~"resolved name in the value namespace to a \
                              set of name bindings with no def?!");
                    }
                    Some(def) => {
                        match def.def {
                            def @ def_variant(*) | def @ def_struct(*) => {
                                return FoundStructOrEnumVariant(def);
                            }
                            def @ def_const(*) => {
                                return FoundConst(def);
                            }
                            _ => {
                                return BareIdentifierPatternUnresolved;
                            }
                        }
                    }
                }
            }

            Indeterminate => {
                fail!(~"unexpected indeterminate result");
            }

            Failed => {
                return BareIdentifierPatternUnresolved;
            }
        }
    }

    /// If `check_ribs` is true, checks the local definitions first; i.e.
    /// doesn't skip straight to the containing module.
    fn resolve_path(@mut self,
                    path: @Path,
                    namespace: Namespace,
                    check_ribs: bool,
                    visitor: ResolveVisitor)
                 -> Option<def> {
        // First, resolve the types.
        for path.types.each |ty| {
            self.resolve_type(*ty, visitor);
        }

        if path.global {
            return self.resolve_crate_relative_path(path,
                                                 self.xray_context,
                                                 namespace);
        }

        if path.idents.len() > 1 {
            return self.resolve_module_relative_path(path,
                                                     self.xray_context,
                                                     namespace);
        }

        return self.resolve_identifier(*path.idents.last(),
                                       namespace,
                                       check_ribs,
                                       path.span);
    }

    fn resolve_identifier(@mut self,
                          identifier: ident,
                          namespace: Namespace,
                          check_ribs: bool,
                          span: span)
                       -> Option<def> {
        if check_ribs {
            match self.resolve_identifier_in_local_ribs(identifier,
                                                      namespace,
                                                      span) {
                Some(def) => {
                    return Some(def);
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
    fn resolve_definition_of_name_in_module(@mut self,
                                            containing_module: @mut Module,
                                            name: ident,
                                            namespace: Namespace,
                                            xray: XrayFlag)
                                         -> NameDefinition {
        // First, search children.
        match containing_module.children.find(&name) {
            Some(child_name_bindings) => {
                match (child_name_bindings.def_for_namespace(namespace),
                       child_name_bindings.privacy_for_namespace(namespace)) {
                    (Some(def), Some(Public)) => {
                        // Found it. Stop the search here.
                        return ChildNameDefinition(def);
                    }
                    (Some(def), _) if xray == Xray => {
                        // Found it. Stop the search here.
                        return ChildNameDefinition(def);
                    }
                    (Some(_), _) | (None, _) => {
                        // Continue.
                    }
                }
            }
            None => {
                // Continue.
            }
        }

        // Next, search import resolutions.
        match containing_module.import_resolutions.find(&name) {
            Some(import_resolution) if import_resolution.privacy == Public ||
                                       xray == Xray => {
                match (*import_resolution).target_for_namespace(namespace) {
                    Some(target) => {
                        match (target.bindings.def_for_namespace(namespace),
                               target.bindings.privacy_for_namespace(
                                    namespace)) {
                            (Some(def), Some(Public)) => {
                                // Found it.
                                import_resolution.state.used = true;
                                return ImportNameDefinition(def);
                            }
                            (Some(_), _) | (None, _) => {
                                // This can happen with external impls, due to
                                // the imperfect way we read the metadata.
                            }
                        }
                    }
                    None => {}
                }
            }
            Some(_) | None => {}    // Continue.
        }

        // Finally, search through external children.
        if namespace == TypeNS {
            match containing_module.external_module_children.find(&name) {
                None => {}
                Some(module) => {
                    match module.def_id {
                        None => {} // Continue.
                        Some(def_id) => {
                            return ChildNameDefinition(def_mod(def_id));
                        }
                    }
                }
            }
        }

        return NoNameDefinition;
    }

    fn intern_module_part_of_path(@mut self, path: @Path) -> ~[ident] {
        let mut module_path_idents = ~[];
        for path.idents.eachi |index, ident| {
            if index == path.idents.len() - 1 {
                break;
            }

            module_path_idents.push(*ident);
        }

        return module_path_idents;
    }

    fn resolve_module_relative_path(@mut self,
                                    path: @Path,
                                    xray: XrayFlag,
                                    namespace: Namespace)
                                 -> Option<def> {
        let module_path_idents = self.intern_module_part_of_path(path);

        let containing_module;
        match self.resolve_module_path_for_import(self.current_module,
                                                  module_path_idents,
                                                  UseLexicalScope,
                                                  path.span) {
            Failed => {
                self.session.span_err(path.span,
                                      fmt!("use of undeclared module `%s`",
                                           self.idents_to_str(
                                               module_path_idents)));
                return None;
            }

            Indeterminate => {
                fail!(~"indeterminate unexpected");
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = *path.idents.last();
        match self.resolve_definition_of_name_in_module(containing_module,
                                                        name,
                                                        namespace,
                                                        xray) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                return None;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) => {
                return Some(def);
            }
        }
    }

    /// Invariant: This must be called only during main resolution, not during
    /// import resolution.
    fn resolve_crate_relative_path(@mut self,
                                   path: @Path,
                                   xray: XrayFlag,
                                   namespace: Namespace)
                                -> Option<def> {
        let module_path_idents = self.intern_module_part_of_path(path);

        let root_module = self.graph_root.get_module();

        let containing_module;
        match self.resolve_module_path_from_root(root_module,
                                                 module_path_idents,
                                                 0,
                                                 path.span,
                                                 SearchItemsAndAllImports) {
            Failed => {
                self.session.span_err(path.span,
                                      fmt!("use of undeclared module `::%s`",
                                            self.idents_to_str(
                                              module_path_idents)));
                return None;
            }

            Indeterminate => {
                fail!(~"indeterminate unexpected");
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = *path.idents.last();
        match self.resolve_definition_of_name_in_module(containing_module,
                                                        name,
                                                        namespace,
                                                        xray) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                return None;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) => {
                return Some(def);
            }
        }
    }

    fn resolve_identifier_in_local_ribs(@mut self,
                                        ident: ident,
                                        namespace: Namespace,
                                        span: span)
                                     -> Option<def> {
        // Check the local set of ribs.
        let search_result;
        match namespace {
            ValueNS => {
                search_result = self.search_ribs(&mut self.value_ribs, ident,
                                                 span,
                                                 DontAllowCapturingSelf);
            }
            TypeNS => {
                search_result = self.search_ribs(&mut self.type_ribs, ident,
                                                 span, AllowCapturingSelf);
            }
        }

        match search_result {
            Some(dl_def(def)) => {
                debug!("(resolving path in local ribs) resolved `%s` to \
                        local: %?",
                       *self.session.str_of(ident),
                       def);
                return Some(def);
            }
            Some(dl_field) | Some(dl_impl(_)) | None => {
                return None;
            }
        }
    }

    fn resolve_item_by_identifier_in_lexical_scope(@mut self,
                                                   ident: ident,
                                                   namespace: Namespace)
                                                -> Option<def> {
        // Check the items.
        match self.resolve_item_in_lexical_scope(self.current_module,
                                                 ident,
                                                 namespace,
                                                 DontSearchThroughModules) {
            Success(target) => {
                match (*target.bindings).def_for_namespace(namespace) {
                    None => {
                        // This can happen if we were looking for a type and
                        // found a module instead. Modules don't have defs.
                        return None;
                    }
                    Some(def) => {
                        debug!("(resolving item path in lexical scope) \
                                resolved `%s` to item",
                               *self.session.str_of(ident));
                        return Some(def);
                    }
                }
            }
            Indeterminate => {
                fail!(~"unexpected indeterminate result");
            }
            Failed => {
                return None;
            }
        }
    }

    fn find_best_match_for_name(@mut self, name: &str, max_distance: uint) -> Option<~str> {
        let this = &mut *self;

        let mut maybes: ~[~str] = ~[];
        let mut values: ~[uint] = ~[];

        let mut j = this.value_ribs.len();
        while j != 0 {
            j -= 1;
            for this.value_ribs[j].bindings.each_key |&k| {
                vec::push(&mut maybes, copy *this.session.str_of(k));
                vec::push(&mut values, uint::max_value);
            }
        }

        let mut smallest = 0;
        for vec::eachi(maybes) |i, &other| {

            values[i] = str::levdistance(name, other);

            if values[i] <= values[smallest] {
                smallest = i;
            }
        }

        if vec::len(values) > 0 &&
            values[smallest] != uint::max_value &&
            values[smallest] < str::len(name) + 2 &&
            values[smallest] <= max_distance &&
            maybes[smallest] != name.to_owned() {

            Some(vec::swap_remove(&mut maybes, smallest))

        } else {
            None
        }
    }

    fn name_exists_in_scope_struct(@mut self, name: &str) -> bool {
        let this = &mut *self;

        let mut i = this.type_ribs.len();
        while i != 0 {
          i -= 1;
          match this.type_ribs[i].kind {
            MethodRibKind(node_id, _) =>
              for this.crate.node.module.items.each |item| {
                if item.id == node_id {
                  match item.node {
                    item_struct(class_def, _) => {
                      for vec::each(class_def.fields) |field| {
                        match field.node.kind {
                          unnamed_field => {},
                          named_field(ident, _, _) => {
                              if str::eq_slice(*this.session.str_of(ident),
                                               name) {
                                return true
                              }
                            }
                        }
                      }
                    }
                    _ => {}
                  }
                }
            },
          _ => {}
        }
      }
      return false;
    }

    fn resolve_expr(@mut self, expr: @expr, visitor: ResolveVisitor) {
        // First, record candidate traits for this expression if it could
        // result in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.node {
            // The interpretation of paths depends on whether the path has
            // multiple elements in it or not.

            expr_path(path) => {
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.

                match self.resolve_path(path, ValueNS, true, visitor) {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving expr) resolved `%s`",
                               self.idents_to_str(path.idents));
                        self.record_def(expr.id, def);
                    }
                    None => {
                        let wrong_name = self.idents_to_str(
                            path.idents);
                        if self.name_exists_in_scope_struct(wrong_name) {
                            self.session.span_err(expr.span,
                                        fmt!("unresolved name: `%s`. \
                                            Did you mean: `self.%s`?",
                                        wrong_name,
                                        wrong_name));
                        }
                        else {
                            // limit search to 5 to reduce the number
                            // of stupid suggestions
                            match self.find_best_match_for_name(wrong_name, 5) {
                                Some(m) => {
                                    self.session.span_err(expr.span,
                                            fmt!("unresolved name: `%s`. \
                                                Did you mean: `%s`?",
                                                wrong_name, m));
                                }
                                None => {
                                    self.session.span_err(expr.span,
                                            fmt!("unresolved name: `%s`.",
                                                wrong_name));
                                }
                            }
                        }
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_fn_block(ref fn_decl, ref block) => {
                self.resolve_function(FunctionRibKind(expr.id, block.node.id),
                                      Some(fn_decl),
                                      NoTypeParameters,
                                      block,
                                      NoSelfBinding,
                                      visitor);
            }

            expr_struct(path, _, _) => {
                // Resolve the path to the structure it goes to.
                let structs: &mut HashSet<def_id> = &mut self.structs;
                match self.resolve_path(path, TypeNS, false, visitor) {
                    Some(def_ty(class_id)) | Some(def_struct(class_id))
                            if structs.contains(&class_id) => {
                        let class_def = def_struct(class_id);
                        self.record_def(expr.id, class_def);
                    }
                    Some(definition @ def_variant(_, class_id))
                            if structs.contains(&class_id) => {
                        self.record_def(expr.id, definition);
                    }
                    _ => {
                        self.session.span_err(
                            path.span,
                            fmt!("`%s` does not name a structure",
                                 self.idents_to_str(path.idents)));
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_loop(_, Some(label)) => {
                do self.with_label_rib {
                    let this = &mut *self;
                    let def_like = dl_def(def_label(expr.id));
                    let rib = this.label_ribs[this.label_ribs.len() - 1];
                    rib.bindings.insert(label, def_like);

                    visit_expr(expr, (), visitor);
                }
            }

            expr_break(Some(label)) | expr_again(Some(label)) => {
                match self.search_ribs(&mut self.label_ribs, label, expr.span,
                                       DontAllowCapturingSelf) {
                    None =>
                        self.session.span_err(expr.span,
                                              fmt!("use of undeclared label \
                                                   `%s`",
                                                   *self.session.str_of(
                                                       label))),
                    Some(dl_def(def @ def_label(_))) =>
                        self.record_def(expr.id, def),
                    Some(_) =>
                        self.session.span_bug(expr.span,
                                              ~"label wasn't mapped to a \
                                                label def!")
                }
            }

            _ => {
                visit_expr(expr, (), visitor);
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(@mut self, expr: @expr) {
        match expr.node {
            expr_field(_, ident, _) => {
                let traits = self.search_for_traits_containing_method(ident);
                self.trait_map.insert(expr.id, @mut traits);
            }
            expr_method_call(_, ident, _, _, _) => {
                let traits = self.search_for_traits_containing_method(ident);
                self.trait_map.insert(expr.id, @mut traits);
            }
            expr_binary(add, _, _) | expr_assign_op(add, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.add_trait());
            }
            expr_binary(subtract, _, _) | expr_assign_op(subtract, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.sub_trait());
            }
            expr_binary(mul, _, _) | expr_assign_op(mul, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.mul_trait());
            }
            expr_binary(div, _, _) | expr_assign_op(div, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.div_trait());
            }
            expr_binary(rem, _, _) | expr_assign_op(rem, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.rem_trait());
            }
            expr_binary(bitxor, _, _) | expr_assign_op(bitxor, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitxor_trait());
            }
            expr_binary(bitand, _, _) | expr_assign_op(bitand, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitand_trait());
            }
            expr_binary(bitor, _, _) | expr_assign_op(bitor, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitor_trait());
            }
            expr_binary(shl, _, _) | expr_assign_op(shl, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.shl_trait());
            }
            expr_binary(shr, _, _) | expr_assign_op(shr, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.shr_trait());
            }
            expr_binary(lt, _, _) | expr_binary(le, _, _) |
            expr_binary(ge, _, _) | expr_binary(gt, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.ord_trait());
            }
            expr_binary(eq, _, _) | expr_binary(ne, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.eq_trait());
            }
            expr_unary(neg, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.neg_trait());
            }
            expr_unary(not, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.not_trait());
            }
            expr_index(*) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.index_trait());
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn search_for_traits_containing_method(@mut self,
                                           name: ident)
                                        -> ~[def_id] {
        debug!("(searching for traits containing method) looking for '%s'",
               *self.session.str_of(name));

        let mut found_traits = ~[];
        let mut search_module = self.current_module;
        loop {
            // Look for the current trait.
            match /*bad*/copy self.current_trait_refs {
                Some(trait_def_ids) => {
                    for trait_def_ids.each |trait_def_id| {
                        self.add_trait_info_if_containing_method(
                            &mut found_traits, *trait_def_id, name);
                    }
                }
                None => {
                    // Nothing to do.
                }
            }

            // Look for trait children.
            for search_module.children.each_value |&child_name_bindings| {
                match child_name_bindings.def_for_namespace(TypeNS) {
                    Some(def) => {
                        match def {
                            def_trait(trait_def_id) => {
                                self.add_trait_info_if_containing_method(
                                    &mut found_traits, trait_def_id, name);
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
            for search_module.import_resolutions.each_value
                    |&import_resolution| {

                match import_resolution.target_for_namespace(TypeNS) {
                    None => {
                        // Continue.
                    }
                    Some(target) => {
                        match target.bindings.def_for_namespace(TypeNS) {
                            Some(def) => {
                                match def {
                                    def_trait(trait_def_id) => {
                                        let added = self.
                                        add_trait_info_if_containing_method(
                                            &mut found_traits,
                                            trait_def_id, name);
                                        if added {
                                            import_resolution.state.used =
                                                true;
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
        }

        return found_traits;
    }

    fn add_trait_info_if_containing_method(&self,
                                           found_traits: &mut ~[def_id],
                                           trait_def_id: def_id,
                                           name: ident)
                                        -> bool {
        debug!("(adding trait info if containing method) trying trait %d:%d \
                for method '%s'",
               trait_def_id.crate,
               trait_def_id.node,
               *self.session.str_of(name));

        match self.trait_info.find(&trait_def_id) {
            Some(trait_info) if trait_info.contains(&name) => {
                debug!("(adding trait info if containing method) found trait \
                        %d:%d for method '%s'",
                       trait_def_id.crate,
                       trait_def_id.node,
                       *self.session.str_of(name));
                found_traits.push(trait_def_id);
                true
            }
            Some(_) | None => {
                false
            }
        }
    }

    fn add_fixed_trait_for_expr(@mut self,
                                expr_id: node_id,
                                trait_id: def_id) {
        self.trait_map.insert(expr_id, @mut ~[trait_id]);
    }

    fn record_def(@mut self, node_id: node_id, def: def) {
        debug!("(recording def) recording %? for %?", def, node_id);
        self.def_map.insert(node_id, def);
    }

    fn enforce_default_binding_mode(@mut self,
                                    pat: @pat,
                                    pat_binding_mode: binding_mode,
                                    descr: &str) {
        match pat_binding_mode {
            bind_infer => {}
            bind_by_copy => {
                self.session.span_err(
                    pat.span,
                    fmt!("cannot use `copy` binding mode with %s",
                         descr));
            }
            bind_by_ref(*) => {
                self.session.span_err(
                    pat.span,
                    fmt!("cannot use `ref` binding mode with %s",
                         descr));
            }
        }
    }

    //
    // main function checking
    //
    // be sure that there is only one main function
    //
    fn check_duplicate_main(@mut self) {
        let this = &mut *self;
        if this.attr_main_fn.is_none() && this.start_fn.is_none() {
            if this.main_fns.len() >= 1u {
                let mut i = 1u;
                while i < this.main_fns.len() {
                    let (_, dup_main_span) = this.main_fns[i].unwrap();
                    this.session.span_err(
                        dup_main_span,
                        ~"multiple 'main' functions");
                    i += 1;
                }
                *this.session.entry_fn = this.main_fns[0];
                *this.session.entry_type = Some(session::EntryMain);
            }
        } else if !this.start_fn.is_none() {
            *this.session.entry_fn = this.start_fn;
            *this.session.entry_type = Some(session::EntryStart);
        } else {
            *this.session.entry_fn = this.attr_main_fn;
            *this.session.entry_type = Some(session::EntryMain);
        }
    }

    //
    // Unused import checking
    //
    // Although this is a lint pass, it lives in here because it depends on
    // resolve data structures.
    //

    fn unused_import_lint_level(@mut self, m: @mut Module) -> level {
        let settings = self.session.lint_settings;
        match m.def_id {
            Some(def) => get_lint_settings_level(settings, unused_imports,
                                                 def.node, def.node),
            None => get_lint_level(settings.default_settings, unused_imports)
        }
    }

    fn check_for_unused_imports_if_necessary(@mut self) {
        if self.unused_import_lint_level(self.current_module) == allow {
            return;
        }

        let root_module = self.graph_root.get_module();
        self.check_for_unused_imports_in_module_subtree(root_module);
    }

    fn check_for_unused_imports_in_module_subtree(@mut self,
                                                  module_: @mut Module) {
        // If this isn't a local crate, then bail out. We don't need to check
        // for unused imports in external crates.

        match module_.def_id {
            Some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
            }
            None => {
                // Check for unused imports in the root module.
            }
            Some(_) => {
                // Bail out.
                debug!("(checking for unused imports in module subtree) not \
                        checking for unused imports for `%s`",
                       self.module_to_str(module_));
                return;
            }
        }

        self.check_for_unused_imports_in_module(module_);

        for module_.children.each_value |&child_name_bindings| {
            match (*child_name_bindings).get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.check_for_unused_imports_in_module_subtree
                        (child_module);
                }
            }
        }

        for module_.anonymous_children.each_value |&child_module| {
            self.check_for_unused_imports_in_module_subtree(child_module);
        }
    }

    fn check_for_unused_imports_in_module(@mut self, module_: @mut Module) {
        for module_.import_resolutions.each_value |&import_resolution| {
            // Ignore dummy spans for things like automatically injected
            // imports for the prelude, and also don't warn about the same
            // import statement being unused more than once. Furthermore, if
            // the import is public, then we can't be sure whether it's unused
            // or not so don't warn about it.
            if !import_resolution.state.used &&
                    !import_resolution.state.warned &&
                    import_resolution.span != dummy_sp() &&
                    import_resolution.privacy != Public {
                import_resolution.state.warned = true;
                let span = import_resolution.span;
                self.session.span_lint_level(
                    self.unused_import_lint_level(module_),
                    span,
                    ~"unused import");
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
    fn module_to_str(@mut self, module_: @mut Module) -> ~str {
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
        return self.idents_to_str(vec::reversed(idents));
    }

    fn dump_module(@mut self, module_: @mut Module) {
        debug!("Dump of module `%s`:", self.module_to_str(module_));

        debug!("Children:");
        for module_.children.each_key |&name| {
            debug!("* %s", *self.session.str_of(name));
        }

        debug!("Import resolutions:");
        for module_.import_resolutions.each |name, import_resolution| {
            let mut value_repr;
            match import_resolution.target_for_namespace(ValueNS) {
                None => { value_repr = ~""; }
                Some(_) => {
                    value_repr = ~" value:?";
                    // FIXME #4954
                }
            }

            let mut type_repr;
            match import_resolution.target_for_namespace(TypeNS) {
                None => { type_repr = ~""; }
                Some(_) => {
                    type_repr = ~" type:?";
                    // FIXME #4954
                }
            }

            debug!("* %s:%s%s", *self.session.str_of(*name),
                   value_repr, type_repr);
        }
    }
}

pub struct CrateMap {
    def_map: DefMap,
    exp_map2: ExportMap2,
    trait_map: TraitMap
}

/// Entry point to crate resolution.
pub fn resolve_crate(session: Session,
                     lang_items: LanguageItems,
                     crate: @crate)
                  -> CrateMap {
    let resolver = @mut Resolver(session, lang_items, crate);
    resolver.resolve();
    let @Resolver{def_map, export_map2, trait_map, _} = resolver;
    CrateMap {
        def_map: def_map,
        exp_map2: export_map2,
        trait_map: trait_map
    }
}
