use driver::session::Session;
use metadata::csearch::{each_path, get_method_names_if_trait};
use metadata::cstore::find_use_stmt_cnum;
use metadata::decoder::{def_like, dl_def, dl_field, dl_impl};
use middle::lang_items::LanguageItems;
use middle::lint::{deny, allow, forbid, level, unused_imports, warn};
use middle::pat_util::{pat_bindings};
use syntax::ast::{_mod, add, arm};
use syntax::ast::{bind_by_ref, bind_by_implicit_ref, bind_by_value};
use syntax::ast::{bitand, bitor, bitxor};
use syntax::ast::{blk, bound_const, bound_copy, bound_owned, bound_send};
use syntax::ast::{bound_trait, binding_mode, capture_clause, class_ctor};
use syntax::ast::{class_dtor, crate, crate_num, decl_item};
use syntax::ast::{def, def_arg, def_binding, def_class, def_const, def_fn};
use syntax::ast::{def_foreign_mod, def_id, def_label, def_local, def_mod};
use syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
use syntax::ast::{def_typaram_binder, def_static_method};
use syntax::ast::{def_upvar, def_use, def_variant, expr, expr_assign_op};
use syntax::ast::{expr_binary, expr_cast, expr_field, expr_fn};
use syntax::ast::{expr_fn_block, expr_index, expr_path};
use syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
use syntax::ast::{def_upvar, def_use, def_variant, div, eq};
use syntax::ast::{enum_variant_kind, expr, expr_again, expr_assign_op};
use syntax::ast::{expr_binary, expr_break, expr_cast, expr_field, expr_fn};
use syntax::ast::{expr_fn_block, expr_index, expr_loop};
use syntax::ast::{expr_path, expr_struct, expr_unary, fn_decl};
use syntax::ast::{foreign_item, foreign_item_const, foreign_item_fn, ge};
use syntax::ast::{gt, ident, impure_fn, inherited, item, item_class};
use syntax::ast::{item_const, item_enum, item_fn, item_foreign_mod};
use syntax::ast::{item_impl, item_mac, item_mod, item_trait, item_ty, le};
use syntax::ast::{local, local_crate, lt, method, module_ns, mul, ne, neg};
use syntax::ast::{node_id, pat, pat_enum, pat_ident, path, prim_ty};
use syntax::ast::{pat_box, pat_lit, pat_range, pat_rec, pat_struct};
use syntax::ast::{pat_tup, pat_uniq, pat_wild, private, provided, public};
use syntax::ast::{required, rem, self_ty_, shl, shr, stmt_decl};
use syntax::ast::{struct_field, struct_variant_kind, sty_static, subtract};
use syntax::ast::{trait_ref, tuple_variant_kind, Ty, ty_bool, ty_char};
use syntax::ast::{ty_f, ty_f32, ty_f64, ty_float, ty_i, ty_i16, ty_i32};
use syntax::ast::{ty_i64, ty_i8, ty_int, ty_param, ty_path, ty_str, ty_u};
use syntax::ast::{ty_u16, ty_u32, ty_u64, ty_u8, ty_uint, type_value_ns};
use syntax::ast::{variant, view_item, view_item_export, view_item_import};
use syntax::ast::{view_item_use, view_path_glob, view_path_list};
use syntax::ast::{view_path_simple, visibility, anonymous, named};
use syntax::ast_util::{def_id_of_def, dummy_sp, local_def};
use syntax::ast_util::{path_to_ident, walk_pat, trait_method_to_ty_method};
use syntax::attr::{attr_metas, contains_name};
use syntax::print::pprust::{pat_to_str, path_to_str};
use syntax::codemap::span;
use syntax::visit::{default_visitor, fk_method, mk_vt, visit_block};
use syntax::visit::{visit_crate, visit_expr, visit_expr_opt, visit_fn};
use syntax::visit::{visit_foreign_item, visit_item, visit_method_helper};
use syntax::visit::{visit_mod, visit_ty, vt};

use box::ptr_eq;
use dvec::DVec;
use option::{get, is_some};
use str::{connect, split_str};
use vec::pop;
use syntax::parse::token::ident_interner;

use std::list::{Cons, List, Nil};
use std::map::HashMap;
use str_eq = str::eq;

// Definition mapping
type DefMap = HashMap<node_id,def>;

struct binding_info {
    span: span,
    binding_mode: binding_mode,
}

// Map from the name in a pattern to its binding mode.
type BindingMap = HashMap<ident,binding_info>;

// Implementation resolution
//
// XXX: This kind of duplicates information kept in ty::method. Maybe it
// should go away.

type MethodInfo = {
    did: def_id,
    n_tps: uint,
    ident: ident,
    self_type: self_ty_
};

type Impl = { did: def_id, ident: ident, methods: ~[@MethodInfo] };

// Trait method resolution
type TraitMap = @HashMap<node_id,@DVec<def_id>>;

// This is the replacement export map. It maps a module to all of the exports
// within.
type ExportMap2 = HashMap<node_id, ~[Export2]>;

struct Export2 {
    name: ~str,         // The name of the target.
    def_id: def_id,     // The definition of the target.
    reexport: bool,     // Whether this is a reexport.
}

enum PatternBindingMode {
    RefutableMode,
    IrrefutableMode
}

impl PatternBindingMode : cmp::Eq {
    pure fn eq(other: &PatternBindingMode) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &PatternBindingMode) -> bool { !self.eq(other) }
}


enum Namespace {
    TypeNS,
    ValueNS
}

enum NamespaceResult {
    UnknownResult,
    UnboundResult,
    BoundResult(@Module, @NameBindings)
}

impl NamespaceResult {
    pure fn is_unknown() -> bool {
        match self {
            UnknownResult => true,
            _ => false
        }
    }
}

enum NameDefinition {
    NoNameDefinition,           //< The name was unbound.
    ChildNameDefinition(def),   //< The name identifies an immediate child.
    ImportNameDefinition(def)   //< The name identifies an import.

}

enum Mutability {
    Mutable,
    Immutable
}

impl Mutability : cmp::Eq {
    pure fn eq(other: &Mutability) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &Mutability) -> bool { !self.eq(other) }
}

enum SelfBinding {
    NoSelfBinding,
    HasSelfBinding(node_id)
}

enum CaptureClause {
    NoCaptureClause,
    HasCaptureClause(capture_clause)
}

type ResolveVisitor = vt<()>;

enum ImportDirectiveNS {
    TypeNSOnly,
    AnyNS
}

impl ImportDirectiveNS : cmp::Eq {
    pure fn eq(other: &ImportDirectiveNS) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &ImportDirectiveNS) -> bool { !self.eq(other) }
}

/// Contains data for specific types of import directives.
enum ImportDirectiveSubclass {
    SingleImport(ident /* target */, ident /* source */, ImportDirectiveNS),
    GlobImport
}

/// The context that we thread through while building the reduced graph.
enum ReducedGraphParent {
    ModuleReducedGraphParent(@Module)
}

enum ResolveResult<T> {
    Failed,         // Failed to resolve the name.
    Indeterminate,  // Couldn't determine due to unresolved globs.
    Success(T)      // Successfully resolved the import.
}

impl<T> ResolveResult<T> {
    fn failed() -> bool {
        match self { Failed => true, _ => false }
    }
    fn indeterminate() -> bool {
        match self { Indeterminate => true, _ => false }
    }
}

enum TypeParameters/& {
    NoTypeParameters,               //< No type parameters.
    HasTypeParameters(&~[ty_param], //< Type parameters.
                      node_id,      //< ID of the enclosing item

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
    FunctionRibKind(node_id /* func id */, node_id /* body id */),

    // We passed through a class, impl, or trait and are now in one of its
    // methods. Allow references to ty params that that class, impl or trait
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
enum MethodSort {
    Required,
    Provided(node_id)
}

// The X-ray flag indicates that a context has the X-ray privilege, which
// allows it to reference private names. Currently, this is used for the test
// runner.
//
// XXX: The X-ray flag is kind of questionable in the first place. It might
// be better to introduce an expr_xray_path instead.

enum XrayFlag {
    NoXray,     //< Private items cannot be accessed.
    Xray        //< Private items can be accessed.
}

impl XrayFlag : cmp::Eq {
    pure fn eq(other: &XrayFlag) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &XrayFlag) -> bool { !self.eq(other) }
}

enum AllowCapturingSelfFlag {
    AllowCapturingSelf,         //< The "self" definition can be captured.
    DontAllowCapturingSelf,     //< The "self" definition cannot be captured.
}

impl AllowCapturingSelfFlag : cmp::Eq {
    pure fn eq(other: &AllowCapturingSelfFlag) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &AllowCapturingSelfFlag) -> bool { !self.eq(other) }
}

enum EnumVariantOrConstResolution {
    FoundEnumVariant(def),
    FoundConst,
    EnumVariantOrConstNotFound
}

// Specifies how duplicates should be handled when adding a child item if
// another item exists with the same name in some namespace.
enum DuplicateCheckingMode {
    ForbidDuplicateModules,
    ForbidDuplicateTypes,
    ForbidDuplicateValues,
    ForbidDuplicateTypesAndValues,
    OverwriteDuplicates
}

impl DuplicateCheckingMode : cmp::Eq {
    pure fn eq(other: &DuplicateCheckingMode) -> bool {
        (self as uint) == (*other as uint)
    }
    pure fn ne(other: &DuplicateCheckingMode) -> bool { !self.eq(other) }
}

// Returns the namespace associated with the given duplicate checking mode,
// or fails for OverwriteDuplicates. This is used for error messages.
fn namespace_for_duplicate_checking_mode(mode: DuplicateCheckingMode) ->
        Namespace {
    match mode {
        ForbidDuplicateModules | ForbidDuplicateTypes |
        ForbidDuplicateTypesAndValues => TypeNS,
        ForbidDuplicateValues => ValueNS,
        OverwriteDuplicates => fail ~"OverwriteDuplicates has no namespace"
    }
}

/// One local scope.
struct Rib {
    bindings: HashMap<ident,def_like>,
    kind: RibKind,
}

fn Rib(kind: RibKind) -> Rib {
    Rib {
        bindings: HashMap(),
        kind: kind
    }
}


/// One import directive.
struct ImportDirective {
    privacy: Privacy,
    module_path: @DVec<ident>,
    subclass: @ImportDirectiveSubclass,
    span: span,
}

fn ImportDirective(privacy: Privacy,
                   module_path: @DVec<ident>,
                   subclass: @ImportDirectiveSubclass,
                   span: span) -> ImportDirective {
    ImportDirective {
        privacy: privacy,
        module_path: module_path,
        subclass: subclass,
        span: span
    }
}

/// The item that an import resolves to.
struct Target {
    target_module: @Module,
    bindings: @NameBindings,
}

fn Target(target_module: @Module, bindings: @NameBindings) -> Target {
    Target {
        target_module: target_module,
        bindings: bindings
    }
}

struct ImportResolution {
    privacy: Privacy,
    span: span,

    // The number of outstanding references to this name. When this reaches
    // zero, outside modules can count on the targets being correct. Before
    // then, all bets are off; future imports could override this name.

    mut outstanding_references: uint,

    mut value_target: Option<Target>,
    mut type_target: Option<Target>,

    mut used: bool,
}

fn ImportResolution(privacy: Privacy, span: span) -> ImportResolution {
    ImportResolution {
        privacy: privacy,
        span: span,
        outstanding_references: 0u,
        value_target: None,
        type_target: None,
        used: false
    }
}

impl ImportResolution {
    fn target_for_namespace(namespace: Namespace) -> Option<Target> {
        match namespace {
            TypeNS      => return copy self.type_target,
            ValueNS     => return copy self.value_target
        }
    }
}

/// The link from a module up to its nearest parent node.
enum ParentLink {
    NoParentLink,
    ModuleParentLink(@Module, ident),
    BlockParentLink(@Module, node_id)
}

/// One node in the tree of modules.
struct Module {
    parent_link: ParentLink,
    mut def_id: Option<def_id>,

    children: HashMap<ident,@NameBindings>,
    imports: DVec<@ImportDirective>,

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

    anonymous_children: HashMap<node_id,@Module>,

    // XXX: This is about to be reworked so that exports are on individual
    // items, not names.
    //
    // The ident is the name of the exported item, while the node ID is the
    // ID of the export path.

    exported_names: HashMap<ident,node_id>,

    // XXX: This is a transition measure to let us switch export-evaluation
    // logic when compiling modules that have transitioned to listing their
    // pub/priv qualifications on items, explicitly, rather than using the
    // old export rule.

    legacy_exports: bool,

    // The status of resolving each import in this module.
    import_resolutions: HashMap<ident,@ImportResolution>,

    // The number of unresolved globs that this module exports.
    mut glob_count: uint,

    // The index of the import we're resolving.
    mut resolved_import_count: uint,
}

fn Module(parent_link: ParentLink,
          def_id: Option<def_id>,
          legacy_exports: bool) -> Module {
    Module {
        parent_link: parent_link,
        def_id: def_id,
        children: HashMap(),
        imports: DVec(),
        anonymous_children: HashMap(),
        exported_names: HashMap(),
        legacy_exports: legacy_exports,
        import_resolutions: HashMap(),
        glob_count: 0u,
        resolved_import_count: 0u
    }
}

impl Module {
    fn all_imports_resolved() -> bool {
        return self.imports.len() == self.resolved_import_count;
    }
}

// XXX: This is a workaround due to is_none in the standard library mistakenly
// requiring a T:copy.

pure fn is_none<T>(x: Option<T>) -> bool {
    match x {
        None => return true,
        Some(_) => return false
    }
}

fn unused_import_lint_level(session: Session) -> level {
    for session.opts.lint_opts.each |lint_option_pair| {
        let (lint_type, lint_level) = *lint_option_pair;
        if lint_type == unused_imports {
            return lint_level;
        }
    }
    return allow;
}

enum Privacy {
    Private,
    Public
}

impl Privacy : cmp::Eq {
    pure fn eq(other: &Privacy) -> bool {
        (self as uint) == ((*other) as uint)
    }
    pure fn ne(other: &Privacy) -> bool { !self.eq(other) }
}

// Records a possibly-private type definition.
struct TypeNsDef {
    mut privacy: Privacy,
    mut module_def: Option<@Module>,
    mut type_def: Option<def>
}

// Records a possibly-private value definition.
struct ValueNsDef {
    privacy: Privacy,
    def: def,
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
struct NameBindings {
    mut type_def: Option<TypeNsDef>,    //< Meaning in type namespace.
    mut value_def: Option<ValueNsDef>,  //< Meaning in value namespace.

    // For error reporting
    // FIXME (#3783): Merge me into TypeNsDef and ValueNsDef.
    mut type_span: Option<span>,
    mut value_span: Option<span>,
}

impl NameBindings {

    /// Creates a new module in this set of name bindings.
    fn define_module(privacy: Privacy,
                     parent_link: ParentLink,
                     def_id: Option<def_id>,
                     legacy_exports: bool,
                     sp: span) {
        // Merges the module with the existing type def or creates a new one.
        let module_ = @Module(parent_link, def_id, legacy_exports);
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
    fn define_type(privacy: Privacy, def: def, sp: span) {
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
    fn define_value(privacy: Privacy, def: def, sp: span) {
        self.value_def = Some(ValueNsDef { privacy: privacy, def: def });
        self.value_span = Some(sp);
    }

    /// Returns the module node if applicable.
    fn get_module_if_available() -> Option<@Module> {
        match self.type_def {
            Some(type_def) => type_def.module_def,
            None => None
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module() -> @Module {
        match self.get_module_if_available() {
            None => {
                fail ~"get_module called on a node with no module \
                       definition!"
            }
            Some(module_def) => module_def
        }
    }

    fn defined_in_namespace(namespace: Namespace) -> bool {
        match namespace {
            TypeNS   => return self.type_def.is_some(),
            ValueNS  => return self.value_def.is_some()
        }
    }

    fn def_for_namespace(namespace: Namespace) -> Option<def> {
        match namespace {
            TypeNS => {
                match self.type_def {
                    None => None,
                    Some(type_def) => {
                        // FIXME (#3784): This is reallllly questionable.
                        // Perhaps the right thing to do is to merge def_mod
                        // and def_ty.
                        match type_def.type_def {
                            Some(type_def) => Some(type_def),
                            None => {
                                match type_def.module_def {
                                    Some(module_def) => {
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

    fn privacy_for_namespace(namespace: Namespace) -> Option<Privacy> {
        match namespace {
            TypeNS => {
                match self.type_def {
                    None => None,
                    Some(type_def) => Some(type_def.privacy)
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

    fn span_for_namespace(namespace: Namespace) -> Option<span> {
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

fn NameBindings() -> NameBindings {
    NameBindings {
        type_def: None,
        value_def: None,
        type_span: None,
        value_span: None
    }
}


/// Interns the names of the primitive types.
struct PrimitiveTypeTable {
    primitive_types: HashMap<ident,prim_ty>,
}

impl PrimitiveTypeTable {
    fn intern(intr: @ident_interner, string: @~str,
              primitive_type: prim_ty) {
        let ident = intr.intern(string);
        self.primitive_types.insert(ident, primitive_type);
    }
}

fn PrimitiveTypeTable(intr: @ident_interner) -> PrimitiveTypeTable {
    let table = PrimitiveTypeTable {
        primitive_types: HashMap()
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


fn namespace_to_str(ns: Namespace) -> ~str {
    match ns {
        TypeNS  => ~"type",
        ValueNS => ~"value",
    }
}

fn has_legacy_export_attr(attrs: &[syntax::ast::attribute]) -> bool {
    for attrs.each |attribute| {
        match attribute.node.value.node {
          syntax::ast::meta_word(w) if w == ~"legacy_exports" => {
            return true;
          }
          _ => {}
        }
    }
    return false;
}

fn Resolver(session: Session, lang_items: LanguageItems,
            crate: @crate) -> Resolver {
    let graph_root = @NameBindings();

    (*graph_root).define_module(Public,
                                NoParentLink,
                                Some({ crate: 0, node: 0 }),
                                has_legacy_export_attr(crate.node.attrs),
                                crate.span);

    let current_module = (*graph_root).get_module();

    let self = Resolver {
        session: session,
        lang_items: copy lang_items,
        crate: crate,

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        graph_root: graph_root,

        unused_import_lint_level: unused_import_lint_level(session),

        trait_info: HashMap(),
        structs: HashMap(),

        unresolved_imports: 0u,

        current_module: current_module,
        value_ribs: @DVec(),
        type_ribs: @DVec(),
        label_ribs: @DVec(),

        xray_context: NoXray,
        current_trait_refs: None,

        self_ident: syntax::parse::token::special_idents::self_,
        primitive_type_table: @PrimitiveTypeTable(session.
                                                  parse_sess.interner),

        namespaces: ~[ TypeNS, ValueNS ],

        def_map: HashMap(),
        export_map2: HashMap(),
        trait_map: @HashMap(),

        intr: session.intr()
    };

    move self
}

/// The main resolver class.
struct Resolver {
    session: Session,
    lang_items: LanguageItems,
    crate: @crate,

    intr: @ident_interner,

    graph_root: @NameBindings,

    unused_import_lint_level: level,

    trait_info: HashMap<def_id,@HashMap<ident,()>>,
    structs: HashMap<def_id,()>,

    // The number of imports that are currently unresolved.
    mut unresolved_imports: uint,

    // The module that represents the current item scope.
    mut current_module: @Module,

    // The current set of local scopes, for values.
    // XXX: Reuse ribs to avoid allocation.
    value_ribs: @DVec<@Rib>,

    // The current set of local scopes, for types.
    type_ribs: @DVec<@Rib>,

    // The current set of local scopes, for labels.
    label_ribs: @DVec<@Rib>,

    // Whether the current context is an X-ray context. An X-ray context is
    // allowed to access private names of any module.
    mut xray_context: XrayFlag,

    // The trait that the current context can refer to.
    mut current_trait_refs: Option<@DVec<def_id>>,

    // The ident for the keyword "self".
    self_ident: ident,

    // The idents for the primitive types.
    primitive_type_table: @PrimitiveTypeTable,

    // The four namespaces.
    namespaces: ~[Namespace],

    def_map: DefMap,
    export_map2: ExportMap2,
    trait_map: TraitMap,
}

impl Resolver {

    /// The main name resolution procedure.
    fn resolve(@self, this: @Resolver) {
        self.build_reduced_graph(this);
        self.session.abort_if_errors();

        self.resolve_imports();
        self.session.abort_if_errors();

        self.record_exports();
        self.session.abort_if_errors();

        self.resolve_crate();
        self.session.abort_if_errors();

        self.check_for_unused_imports_if_necessary();
    }

    //
    // Reduced graph building
    //
    // Here we build the "reduced graph": the graph of the module tree without
    // any imports resolved.
    //

    /// Constructs the reduced graph for the entire crate.
    fn build_reduced_graph(this: @Resolver) {
        let initial_parent =
            ModuleReducedGraphParent((*self.graph_root).get_module());
        visit_crate(*self.crate, initial_parent, mk_vt(@{
            visit_item: |item, context, visitor|
                (*this).build_reduced_graph_for_item(item, context, visitor),

            visit_foreign_item: |foreign_item, context, visitor|
                (*this).build_reduced_graph_for_foreign_item(foreign_item,
                                                             context,
                                                             visitor),

            visit_view_item: |view_item, context, visitor|
                (*this).build_reduced_graph_for_view_item(view_item,
                                                          context,
                                                          visitor),

            visit_block: |block, context, visitor|
                (*this).build_reduced_graph_for_block(block,
                                                      context,
                                                      visitor),

            .. *default_visitor()
        }));
    }

    fn visibility_to_privacy(visibility: visibility,
                             legacy_exports: bool) -> Privacy {
        if legacy_exports {
            match visibility {
              inherited | public => Public,
              private => Private
            }
        } else {
            match visibility {
              public => Public,
              inherited | private => Private
            }
        }
    }

    /// Returns the current module tracked by the reduced graph parent.
    fn get_module_from_parent(reduced_graph_parent: ReducedGraphParent)
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
    fn add_child(name: ident,
                 reduced_graph_parent: ReducedGraphParent,
                 duplicate_checking_mode: DuplicateCheckingMode,
                 // For printing errors
                 sp: span)
              -> (@NameBindings, ReducedGraphParent) {

        // If this is the immediate descendant of a module, then we add the
        // child name directly. Otherwise, we create or reuse an anonymous
        // module and add the child to that.

        let mut module_;
        match reduced_graph_parent {
            ModuleReducedGraphParent(parent_module) => {
                module_ = parent_module;
            }
        }

        // Add or reuse the child.
        let new_parent = ModuleReducedGraphParent(module_);
        match module_.children.find(name) {
            None => {
                let child = @NameBindings();
                module_.children.insert(name, child);
                return (child, new_parent);
            }
            Some(child) => {
                // Enforce the duplicate checking mode. If we're requesting
                // duplicate module checking, check that there isn't a module
                // in the module with the same name. If we're requesting
                // duplicate type checking, check that there isn't a type in
                // the module with the same name. If we're requesting
                // duplicate value checking, check that there isn't a value in
                // the module with the same name. If we're requesting
                // duplicate type checking and duplicate value checking, check
                // that there isn't a duplicate type and a duplicate value
                // with the same name. If no duplicate checking was requested
                // at all, do nothing.

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
                             self.session.str_of(name)));
                    do child.span_for_namespace(ns).iter() |sp| {
                        self.session.span_note(*sp,
                             fmt!("first definition of %s %s here:",
                                  namespace_to_str(ns),
                                  self.session.str_of(name)));
                    }
                }
                return (child, new_parent);
            }
        }
    }

    fn block_needs_anonymous_module(block: blk) -> bool {
        // If the block has view items, we need an anonymous module.
        if block.node.view_items.len() > 0u {
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

    fn get_parent_link(parent: ReducedGraphParent,
                       name: ident) -> ParentLink {
        match parent {
            ModuleReducedGraphParent(module_) => {
                return ModuleParentLink(module_, name);
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(item: @item,
                                    parent: ReducedGraphParent,
                                    &&visitor: vt<ReducedGraphParent>) {

        let ident = item.ident;
        let sp = item.span;
        let legacy = match parent {
          ModuleReducedGraphParent(m) => m.legacy_exports
        };
        let privacy = self.visibility_to_privacy(item.vis, legacy);

        match item.node {
            item_mod(module_) => {
                let legacy = has_legacy_export_attr(item.attrs);
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateModules, sp);

                let parent_link = self.get_parent_link(new_parent, ident);
                let def_id = { crate: 0, node: item.id };
                (*name_bindings).define_module(privacy, parent_link,
                                               Some(def_id), legacy, sp);

                let new_parent =
                    ModuleReducedGraphParent((*name_bindings).get_module());

                visit_mod(module_, sp, item.id, new_parent, visitor);
            }

            item_foreign_mod(fm) => {
                let legacy = has_legacy_export_attr(item.attrs);
                let new_parent = match fm.sort {
                    named => {
                        let (name_bindings, new_parent) =
                            self.add_child(ident, parent,
                                           ForbidDuplicateModules, sp);

                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        let def_id = { crate: 0, node: item.id };
                        (*name_bindings).define_module(privacy,
                                                       parent_link,
                                                       Some(def_id),
                                                       legacy,
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

                (*name_bindings).define_value
                    (privacy, def_const(local_def(item.id)), sp);
            }
            item_fn(_, purity, _, _) => {
              let (name_bindings, new_parent) =
                self.add_child(ident, parent, ForbidDuplicateValues, sp);

                let def = def_fn(local_def(item.id), purity);
                (*name_bindings).define_value(privacy, def, sp);
                visit_item(item, new_parent, visitor);
            }

            // These items live in the type namespace.
            item_ty(*) => {
                let (name_bindings, _) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                (*name_bindings).define_type
                    (privacy, def_ty(local_def(item.id)), sp);
            }

            item_enum(enum_definition, _) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                (*name_bindings).define_type
                    (privacy, def_ty(local_def(item.id)), sp);

                for enum_definition.variants.each |variant| {
                    self.build_reduced_graph_for_variant(*variant,
                                                         local_def(item.id),
                                                         privacy,
                                                         new_parent,
                                                         visitor);
                }
            }

            // These items live in both the type and value namespaces.
            item_class(*) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                (*name_bindings).define_type
                    (privacy, def_ty(local_def(item.id)), sp);

                // Record the def ID of this struct.
                self.structs.insert(local_def(item.id), ());

                visit_item(item, new_parent, visitor);
            }

            item_impl(_, trait_ref_opt, ty, methods) => {
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
                    (None, @{ id: _, node: ty_path(path, _), span: _ }) if
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
                        name_bindings.define_module(privacy, parent_link,
                                                    Some(def_id), false, sp);

                        let new_parent = ModuleReducedGraphParent(
                            name_bindings.get_module());

                        // For each static method...
                        for methods.each |method| {
                            match method.self_ty.node {
                                sty_static => {
                                    // Add the static method to the module.
                                    let ident = method.ident;
                                    let (method_name_bindings, _) =
                                        self.add_child(ident,
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

            item_trait(_, _, methods) => {
                let (name_bindings, new_parent) =
                    self.add_child(ident, parent, ForbidDuplicateTypes, sp);

                // Add the names of all the methods to the trait info.
                let method_names = @HashMap();
                for methods.each |method| {
                    let ty_m = trait_method_to_ty_method(*method);

                    let ident = ty_m.ident;
                    // Add it to the trait info if not static,
                    // add it as a name in the enclosing module otherwise.
                    match ty_m.self_ty.node {
                      sty_static => {
                        // which parent to use??
                        let (method_name_bindings, _) =
                            self.add_child(ident, new_parent,
                                           ForbidDuplicateValues, ty_m.span);
                        let def = def_static_method(local_def(ty_m.id),
                                                    local_def(item.id),
                                                    ty_m.purity);
                        (*method_name_bindings).define_value
                            (Public, def, ty_m.span);
                      }
                      _ => {
                        (*method_names).insert(ident, ());
                      }
                    }
                }

                let def_id = local_def(item.id);
                self.trait_info.insert(def_id, method_names);

                (*name_bindings).define_type
                    (privacy,
                     def_ty(def_id),
                     sp);
                visit_item(item, new_parent, visitor);
            }

            item_mac(*) => {
                fail ~"item macros unimplemented"
            }
        }
    }

    // Constructs the reduced graph for one variant. Variants exist in the
    // type and/or value namespaces.
    fn build_reduced_graph_for_variant(variant: variant,
                                       item_id: def_id,
                                       +parent_privacy: Privacy,
                                       parent: ReducedGraphParent,
                                       &&visitor: vt<ReducedGraphParent>) {

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
                (*child).define_value(privacy,
                                      def_variant(item_id,
                                                  local_def(variant.node.id)),
                                      variant.span);
            }
            struct_variant_kind(_) => {
                (*child).define_type(privacy,
                                     def_variant(item_id,
                                                 local_def(variant.node.id)),
                                     variant.span);
                self.structs.insert(local_def(variant.node.id), ());
            }
            enum_variant_kind(enum_definition) => {
                (*child).define_type(privacy,
                                     def_ty(local_def(variant.node.id)),
                                     variant.span);
                for enum_definition.variants.each |variant| {
                    self.build_reduced_graph_for_variant(*variant, item_id,
                                                         parent_privacy,
                                                         parent, visitor);
                }
            }
        }
    }

    /**
     * Constructs the reduced graph for one 'view item'. View items consist
     * of imports and use directives.
     */
    fn build_reduced_graph_for_view_item(view_item: @view_item,
                                         parent: ReducedGraphParent,
                                         &&_visitor: vt<ReducedGraphParent>) {

        let legacy = match parent {
          ModuleReducedGraphParent(m) => m.legacy_exports
        };
        let privacy = self.visibility_to_privacy(view_item.vis, legacy);
        match view_item.node {
            view_item_import(view_paths) => {
                for view_paths.each |view_path| {
                    // Extract and intern the module part of the path. For
                    // globs and lists, the path is found directly in the AST;
                    // for simple paths we have to munge the path a little.

                    let module_path = @DVec();
                    match view_path.node {
                        view_path_simple(_, full_path, _, _) => {
                            let path_len = full_path.idents.len();
                            assert path_len != 0u;

                            for full_path.idents.eachi |i, ident| {
                                if i != path_len - 1u {
                                    (*module_path).push(*ident);
                                }
                            }
                        }

                        view_path_glob(module_ident_path, _) |
                        view_path_list(module_ident_path, _, _) => {
                            for module_ident_path.idents.each |ident| {
                                (*module_path).push(*ident);
                            }
                        }
                    }

                    // Build up the import directives.
                    let module_ = self.get_module_from_parent(parent);
                    match view_path.node {
                        view_path_simple(binding, full_path, ns, _) => {
                            let ns = match ns {
                                module_ns => TypeNSOnly,
                                type_value_ns => AnyNS
                            };

                            let source_ident = full_path.idents.last();
                            let subclass = @SingleImport(binding,
                                                         source_ident,
                                                         ns);
                            self.build_import_directive(privacy,
                                                        module_,
                                                        module_path,
                                                        subclass,
                                                        view_path.span);
                        }
                        view_path_list(_, source_idents, _) => {
                            for source_idents.each |source_ident| {
                                let name = source_ident.node.name;
                                let subclass = @SingleImport(name,
                                                             name,
                                                             AnyNS);
                                self.build_import_directive(privacy,
                                                            module_,
                                                            module_path,
                                                            subclass,
                                                            view_path.span);
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

            view_item_export(view_paths) => {
                let module_ = self.get_module_from_parent(parent);
                for view_paths.each |view_path| {
                    match view_path.node {
                        view_path_simple(ident, full_path, _, ident_id) => {
                            let last_ident = full_path.idents.last();
                            if last_ident != ident {
                                self.session.span_err(view_item.span,
                                                      ~"cannot export under \
                                                       a new name");
                            }
                            if full_path.idents.len() != 1u {
                                self.session.span_err(
                                    view_item.span,
                                    ~"cannot export an item \
                                      that is not in this \
                                      module");
                            }

                            module_.exported_names.insert(ident, ident_id);
                        }

                        view_path_glob(*) => {
                            self.session.span_err(view_item.span,
                                                  ~"export globs are \
                                                   unsupported");
                        }

                        view_path_list(path, path_list_idents, _) => {
                            if path.idents.len() == 1u &&
                                    path_list_idents.len() == 0u {

                                self.session.span_warn(view_item.span,
                                                       ~"this syntax for \
                                                        exporting no \
                                                        variants is \
                                                        unsupported; export \
                                                        variants \
                                                        individually");
                            } else {
                                if path.idents.len() != 0u {
                                    self.session.span_err(view_item.span,
                                                          ~"cannot export an \
                                                           item that is not \
                                                           in this module");
                                }

                                for path_list_idents.each |path_list_ident| {
                                    let ident = path_list_ident.node.name;
                                    let id = path_list_ident.node.id;
                                    module_.exported_names.insert(ident, id);
                                }
                            }
                        }
                    }
                }
            }

            view_item_use(name, _, node_id) => {
                match find_use_stmt_cnum(self.session.cstore, node_id) {
                    Some(crate_id) => {
                        let (child_name_bindings, new_parent) =
                            self.add_child(name, parent, ForbidDuplicateTypes,
                                           view_item.span);

                        let def_id = { crate: crate_id, node: 0 };
                        let parent_link = ModuleParentLink
                            (self.get_module_from_parent(new_parent), name);

                        (*child_name_bindings).define_module(privacy,
                                                             parent_link,
                                                             Some(def_id),
                                                             false,
                                                             view_item.span);
                        self.build_reduced_graph_for_external_crate
                            ((*child_name_bindings).get_module());
                    }
                    None => {
                        /* Ignore. */
                    }
                }
            }
        }
    }

    /// Constructs the reduced graph for one foreign item.
    fn build_reduced_graph_for_foreign_item(foreign_item: @foreign_item,
                                            parent: ReducedGraphParent,
                                            &&visitor:
                                                vt<ReducedGraphParent>) {

        let name = foreign_item.ident;
        let (name_bindings, new_parent) =
            self.add_child(name, parent, ForbidDuplicateValues,
                           foreign_item.span);

        match foreign_item.node {
            foreign_item_fn(_, purity, type_parameters) => {
                let def = def_fn(local_def(foreign_item.id), purity);
                (*name_bindings).define_value(Public, def, foreign_item.span);

                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, foreign_item.id,
                                           0u, NormalRibKind)) {
                    visit_foreign_item(foreign_item, new_parent, visitor);
                }
            }
            foreign_item_const(*) => {
                let def = def_const(local_def(foreign_item.id));
                (*name_bindings).define_value(Public, def, foreign_item.span);

                visit_foreign_item(foreign_item, new_parent, visitor);
            }
        }
    }

    fn build_reduced_graph_for_block(block: blk,
                                     parent: ReducedGraphParent,
                                     &&visitor: vt<ReducedGraphParent>) {

        let mut new_parent;
        if self.block_needs_anonymous_module(block) {
            let block_id = block.node.id;

            debug!("(building reduced graph for block) creating a new \
                    anonymous module for block %d",
                   block_id);

            let parent_module = self.get_module_from_parent(parent);
            let new_module = @Module(BlockParentLink(parent_module, block_id),
                                     None, false);
            parent_module.anonymous_children.insert(block_id, new_module);
            new_parent = ModuleReducedGraphParent(new_module);
        } else {
            new_parent = parent;
        }

        visit_block(block, new_parent, visitor);
    }

    fn handle_external_def(def: def, modules: HashMap<def_id, @Module>,
                           child_name_bindings: @NameBindings,
                           final_ident: ~str,
                           ident: ident, new_parent: ReducedGraphParent) {
        match def {
          def_mod(def_id) | def_foreign_mod(def_id) => {
            match copy child_name_bindings.type_def {
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

                match modules.find(def_id) {
                  None => {
                    child_name_bindings.define_module(Public,
                                                      parent_link,
                                                      Some(def_id),
                                                      false,
                                                      dummy_sp());
                    modules.insert(def_id,
                                   child_name_bindings.get_module());
                  }
                  Some(existing_module) => {
                    // Create an import resolution to
                    // avoid creating cycles in the
                    // module graph.

                    let resolution = @ImportResolution(Public, dummy_sp());
                    resolution.outstanding_references = 0;

                    match existing_module.parent_link {
                      NoParentLink |
                      BlockParentLink(*) => {
                        fail ~"can't happen";
                      }
                      ModuleParentLink(parent_module, ident) => {
                        let name_bindings = parent_module.children.get(ident);
                        resolution.type_target =
                            Some(Target(parent_module, name_bindings));
                      }
                    }

                    debug!("(building reduced graph for external crate) \
                            ... creating import resolution");

                    new_parent.import_resolutions.insert(ident, resolution);
                  }
                }
              }
            }
          }
          def_fn(*) | def_static_method(*) | def_const(*) |
          def_variant(*) => {
            debug!("(building reduced graph for external \
                    crate) building value %s", final_ident);
            (*child_name_bindings).define_value(Public, def, dummy_sp());
          }
          def_ty(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type %s", final_ident);

            // If this is a trait, add all the method names
            // to the trait info.

            match get_method_names_if_trait(self.session.cstore, def_id) {
              None => {
                // Nothing to do.
              }
              Some(method_names) => {
                let interned_method_names = @HashMap();
                for method_names.each |method_data| {
                    let (method_name, self_ty) = *method_data;
                    debug!("(building reduced graph for \
                            external crate) ... adding \
                            trait method '%s'",
                           self.session.str_of(method_name));

                    // Add it to the trait info if not static.
                    if self_ty != sty_static {
                        interned_method_names.insert(method_name, ());
                    }
                }
                self.trait_info.insert(def_id, interned_method_names);
              }
            }

            child_name_bindings.define_type(Public, def, dummy_sp());
          }
          def_class(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type %s",
                   final_ident);
            child_name_bindings.define_type(Public, def, dummy_sp());
            self.structs.insert(def_id, ());
          }
          def_self(*) | def_arg(*) | def_local(*) |
          def_prim_ty(*) | def_ty_param(*) | def_binding(*) |
          def_use(*) | def_upvar(*) | def_region(*) |
          def_typaram_binder(*) | def_label(*) => {
            fail fmt!("didn't expect `%?`", def);
          }
        }
    }

    /**
     * Builds the reduced graph rooted at the 'use' directive for an external
     * crate.
     */
    fn build_reduced_graph_for_external_crate(root: @Module) {
        let modules = HashMap();

        // Create all the items reachable by paths.
        for each_path(self.session.cstore, root.def_id.get().crate)
                |path_entry| {

            debug!("(building reduced graph for external crate) found path \
                        entry: %s (%?)",
                    path_entry.path_string,
                    path_entry.def_like);

            let mut pieces = split_str(path_entry.path_string, ~"::");
            let final_ident_str = pieces.pop();
            let final_ident = self.session.ident_of(final_ident_str);

            // Find the module we need, creating modules along the way if we
            // need to.

            let mut current_module = root;
            for pieces.each |ident_str| {
                let ident = self.session.ident_of(*ident_str);
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
                                autovivifying %s", *ident_str);
                        let parent_link = self.get_parent_link(new_parent,
                                                               ident);
                        (*child_name_bindings).define_module(Public,
                                                             parent_link,
                                                             None, false,
                                                             dummy_sp());
                    }
                    Some(_) => { /* Fall through. */ }
                }

                current_module = (*child_name_bindings).get_module();
            }

            // Add the new child item.
            let (child_name_bindings, new_parent) =
                self.add_child(final_ident,
                               ModuleReducedGraphParent(current_module),
                               OverwriteDuplicates,
                               dummy_sp());

            match path_entry.def_like {
                dl_def(def) => {
                    self.handle_external_def(def, modules,
                                             child_name_bindings,
                                             self.session.str_of(final_ident),
                                             final_ident, new_parent);
                }
                dl_impl(_) => {
                    // We only process static methods of impls here.
                    debug!("(building reduced graph for external crate) \
                            processing impl %s", final_ident_str);

                    // FIXME (#3786): Cross-crate static methods in anonymous
                    // traits.
                }
                dl_field => {
                    debug!("(building reduced graph for external crate) \
                            ignoring field %s", final_ident_str);
                }
            }
        }
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(privacy: Privacy,
                              module_: @Module,
                              module_path: @DVec<ident>,
                              subclass: @ImportDirectiveSubclass,
                              span: span) {

        let directive = @ImportDirective(privacy, module_path,
                                         subclass, span);
        module_.imports.push(directive);

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match *subclass {
            SingleImport(target, _, _) => {
                debug!("(building import directive) building import \
                        directive: privacy %? %s::%s",
                       privacy,
                       self.idents_to_str(module_path.get()),
                       self.session.str_of(target));

                match module_.import_resolutions.find(target) {
                    Some(resolution) => {
                        debug!("(building import directive) bumping \
                                reference");
                        resolution.outstanding_references += 1u;
                    }
                    None => {
                        debug!("(building import directive) creating new");
                        let resolution = @ImportResolution(privacy, span);
                        resolution.outstanding_references = 1u;
                        module_.import_resolutions.insert(target, resolution);
                    }
                }
            }
            GlobImport => {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module_.glob_count += 1u;
            }
        }

        self.unresolved_imports += 1u;
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
    fn resolve_imports() {
        let mut i = 0u;
        let mut prev_unresolved_imports = 0u;
        loop {
            debug!("(resolving imports) iteration %u, %u imports left",
                   i, self.unresolved_imports);

            let module_root = (*self.graph_root).get_module();
            self.resolve_imports_for_module_subtree(module_root);

            if self.unresolved_imports == 0u {
                debug!("(resolving imports) success");
                break;
            }

            if self.unresolved_imports == prev_unresolved_imports {
                self.session.err(~"failed to resolve imports");
                self.report_unresolved_imports(module_root);
                break;
            }

            i += 1u;
            prev_unresolved_imports = self.unresolved_imports;
        }
    }

    /**
     * Attempts to resolve imports for the given module and all of its
     * submodules.
     */
    fn resolve_imports_for_module_subtree(module_: @Module) {
        debug!("(resolving imports for module subtree) resolving %s",
               self.module_to_str(module_));
        self.resolve_imports_for_module(module_);

        for module_.children.each |_name, child_node| {
            match child_node.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.resolve_imports_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each |_block_id, child_module| {
            self.resolve_imports_for_module_subtree(child_module);
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(module_: @Module) {
        if (*module_).all_imports_resolved() {
            debug!("(resolving imports for module) all imports resolved for \
                   %s",
                   self.module_to_str(module_));
            return;
        }

        let import_count = module_.imports.len();
        while module_.resolved_import_count < import_count {
            let import_index = module_.resolved_import_count;
            let import_directive = module_.imports.get_elt(import_index);
            match self.resolve_import_for_module(module_, import_directive) {
                Failed => {
                    // We presumably emitted an error. Continue.
                    self.session.span_err(import_directive.span,
                                          ~"failed to resolve import");
                }
                Indeterminate => {
                    // Bail out. We'll come around next time.
                    break;
                }
                Success(()) => {
                    // Good. Continue.
                }
            }

            module_.resolved_import_count += 1u;
        }
    }

    fn idents_to_str(idents: ~[ident]) -> ~str {
        // XXX: str::connect should do this.
        let mut result = ~"";
        let mut first = true;
        for idents.each() |ident| {
            if first {
                first = false;
            } else {
                result += ~"::";
            }
            result += self.session.str_of(*ident);
        }
        // XXX: Shouldn't copy here. We need string builder functionality.
        return result;
    }
    /**
     * Attempts to resolve the given import. The return value indicates
     * failure if we're certain the name does not exist, indeterminate if we
     * don't know whether the name exists at the moment due to other
     * currently-unresolved imports, or success if we know the name exists.
     * If successful, the resolved bindings are written into the module.
     */
    fn resolve_import_for_module(module_: @Module,
                                 import_directive: @ImportDirective)
                              -> ResolveResult<()> {

        let mut resolution_result;
        let module_path = import_directive.module_path;

        debug!("(resolving import for module) resolving import `%s::...` in \
                `%s`",
               self.idents_to_str((*module_path).get()),
               self.module_to_str(module_));

        // One-level renaming imports of the form `import foo = bar;` are
        // handled specially.

        if (*module_path).len() == 0u {
            resolution_result =
                self.resolve_one_level_renaming_import(module_,
                                                       import_directive);
        } else {
            // First, resolve the module path for the directive, if necessary.
            match self.resolve_module_path_for_import(module_,
                                                    module_path,
                                                    NoXray,
                                                    import_directive.span) {

                Failed => {
                    resolution_result = Failed;
                }
                Indeterminate => {
                    resolution_result = Indeterminate;
                }
                Success(containing_module) => {
                    // We found the module that the target is contained
                    // within. Attempt to resolve the import within it.

                    match *import_directive.subclass {
                        SingleImport(target, source, AnyNS) => {
                            resolution_result =
                                self.resolve_single_import(module_,
                                                           containing_module,
                                                           target,
                                                           source);
                        }
                        SingleImport(target, source, TypeNSOnly) => {
                            resolution_result =
                                self.resolve_single_module_import
                                    (module_, containing_module, target,
                                     source);
                        }
                        GlobImport => {
                            let span = import_directive.span;
                            let p = import_directive.privacy;
                            resolution_result =
                                self.resolve_glob_import(p, module_,
                                                         containing_module,
                                                         span);
                        }
                    }
                }
            }
        }

        // Decrement the count of unresolved imports.
        match resolution_result {
            Success(()) => {
                assert self.unresolved_imports >= 1u;
                self.unresolved_imports -= 1u;
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
                    assert module_.glob_count >= 1u;
                    module_.glob_count -= 1u;
                }
                SingleImport(*) => {
                    // Ignore.
                }
            }
        }

        return resolution_result;
    }

    fn resolve_single_import(module_: @Module,
                             containing_module: @Module,
                             target: ident,
                             source: ident)
                          -> ResolveResult<()> {

        debug!("(resolving single import) resolving `%s` = `%s::%s` from \
                `%s`",
               self.session.str_of(target),
               self.module_to_str(containing_module),
               self.session.str_of(source),
               self.module_to_str(module_));

        if !self.name_is_exported(containing_module, source) {
            debug!("(resolving single import) name `%s` is unexported",
                   self.session.str_of(source));
            return Failed;
        }

        // We need to resolve both namespaces for this to succeed.
        //
        // XXX: See if there's some way of handling namespaces in a more
        // generic way. We have two of them; it seems worth doing...

        let mut value_result = UnknownResult;
        let mut type_result = UnknownResult;

        // Search for direct children of the containing module.
        match containing_module.children.find(source) {
            None => {
                // Continue.
            }
            Some(child_name_bindings) => {
                if (*child_name_bindings).defined_in_namespace(ValueNS) {
                    value_result = BoundResult(containing_module,
                                               child_name_bindings);
                }
                if (*child_name_bindings).defined_in_namespace(TypeNS) {
                    type_result = BoundResult(containing_module,
                                              child_name_bindings);
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

                if containing_module.glob_count > 0u {
                    debug!("(resolving single import) unresolved glob; \
                            bailing out");
                    return Indeterminate;
                }

                // Now search the exported imports within the containing
                // module.

                match containing_module.import_resolutions.find(source) {
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
                                == 0u => {

                        fn get_binding(import_resolution: @ImportResolution,
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
                                    import_resolution.used = true;
                                    return BoundResult(target.target_module,
                                                    target.bindings);
                                }
                            }
                        }

                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.
                        if value_result.is_unknown() {
                            value_result = get_binding(import_resolution,
                                                       ValueNS);
                        }
                        if type_result.is_unknown() {
                            type_result = get_binding(import_resolution,
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

        // We've successfully resolved the import. Write the results in.
        assert module_.import_resolutions.contains_key(target);
        let import_resolution = module_.import_resolutions.get(target);

        match value_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.value_target =
                    Some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail ~"value result should be known at this point";
            }
        }
        match type_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.type_target =
                    Some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail ~"type result should be known at this point";
            }
        }

        let i = import_resolution;
        match (i.value_target, i.type_target) {
          // If this name wasn't found in either namespace, it's definitely
          // unresolved.
          (None, None) => { return Failed; }
          _ => {}
        }

        assert import_resolution.outstanding_references >= 1u;
        import_resolution.outstanding_references -= 1u;

        debug!("(resolving single import) successfully resolved import");
        return Success(());
    }

    fn resolve_single_module_import(module_: @Module,
                                    containing_module: @Module,
                                    target: ident,
                                    source: ident)
                                 -> ResolveResult<()> {

        debug!("(resolving single module import) resolving `%s` = `%s::%s` \
                from `%s`",
               self.session.str_of(target),
               self.module_to_str(containing_module),
               self.session.str_of(source),
               self.module_to_str(module_));

        if !self.name_is_exported(containing_module, source) {
            debug!("(resolving single import) name `%s` is unexported",
                   self.session.str_of(source));
            return Failed;
        }

        // We need to resolve the module namespace for this to succeed.
        let mut module_result = UnknownResult;

        // Search for direct children of the containing module.
        match containing_module.children.find(source) {
            None => {
                // Continue.
            }
            Some(child_name_bindings) => {
                if (*child_name_bindings).defined_in_namespace(TypeNS) {
                    module_result = BoundResult(containing_module,
                                                child_name_bindings);
                }
            }
        }

        // Unless we managed to find a result, search imports as well.
        match module_result {
            BoundResult(*) => {
                // Continue.
            }
            _ => {
                // If there is an unresolved glob at this point in the
                // containing module, bail out. We don't know enough to be
                // able to resolve this import.

                if containing_module.glob_count > 0u {
                    debug!("(resolving single module import) unresolved \
                            glob; bailing out");
                    return Indeterminate;
                }

                // Now search the exported imports within the containing
                // module.

                match containing_module.import_resolutions.find(source) {
                    None => {
                        // The containing module definitely doesn't have an
                        // exported import with the name in question. We can
                        // therefore accurately report that the names are
                        // unbound.

                        if module_result.is_unknown() {
                            module_result = UnboundResult;
                        }
                    }
                    Some(import_resolution)
                            if import_resolution.outstanding_references
                                == 0u => {
                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.

                        if module_result.is_unknown() {
                            match (*import_resolution).target_for_namespace(
                                    TypeNS) {
                                None => {
                                    module_result = UnboundResult;
                                }
                                Some(target) => {
                                    import_resolution.used = true;
                                    module_result = BoundResult
                                        (target.target_module,
                                         target.bindings);
                                }
                            }
                        }
                    }
                    Some(_) => {
                        // The import is unresolved. Bail out.
                        debug!("(resolving single module import) unresolved \
                                import; bailing out");
                        return Indeterminate;
                    }
                }
            }
        }

        // We've successfully resolved the import. Write the results in.
        assert module_.import_resolutions.contains_key(target);
        let import_resolution = module_.import_resolutions.get(target);

        match module_result {
            BoundResult(target_module, name_bindings) => {
                debug!("(resolving single import) found module binding");
                import_resolution.type_target =
                    Some(Target(target_module, name_bindings));
            }
            UnboundResult => {
                debug!("(resolving single import) didn't find module \
                        binding");
            }
            UnknownResult => {
                fail ~"module result should be known at this point";
            }
        }

        let i = import_resolution;
        if i.type_target.is_none() {
          // If this name wasn't found in the type namespace, it's
          // definitely unresolved.
          return Failed;
        }

        assert import_resolution.outstanding_references >= 1u;
        import_resolution.outstanding_references -= 1u;

        debug!("(resolving single module import) successfully resolved \
               import");
        return Success(());
    }


    /**
     * Resolves a glob import. Note that this function cannot fail; it either
     * succeeds or bails out (as importing * from an empty module or a module
     * that exports nothing is valid).
     */
    fn resolve_glob_import(privacy: Privacy,
                           module_: @Module,
                           containing_module: @Module,
                           span: span)
                        -> ResolveResult<()> {

        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).

        if !(*containing_module).all_imports_resolved() {
            debug!("(resolving glob import) target module has unresolved \
                    imports; bailing out");
            return Indeterminate;
        }

        assert containing_module.glob_count == 0u;

        // Add all resolved imports from the containing module.
        for containing_module.import_resolutions.each
                |ident, target_import_resolution| {

            if !self.name_is_exported(containing_module, ident) {
                debug!("(resolving glob import) name `%s` is unexported",
                       self.session.str_of(ident));
                loop;
            }

            debug!("(resolving glob import) writing module resolution \
                    %? into `%s`",
                   is_none(target_import_resolution.type_target),
                   self.module_to_str(module_));

            // Here we merge two import resolutions.
            match module_.import_resolutions.find(ident) {
                None => {
                    // Simple: just copy the old import resolution.
                    let new_import_resolution =
                        @ImportResolution(privacy,
                                          target_import_resolution.span);
                    new_import_resolution.value_target =
                        copy target_import_resolution.value_target;
                    new_import_resolution.type_target =
                        copy target_import_resolution.type_target;

                    module_.import_resolutions.insert
                        (ident, new_import_resolution);
                }
                Some(dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match copy target_import_resolution.value_target {
                        None => {
                            // Continue.
                        }
                        Some(value_target) => {
                            dest_import_resolution.value_target =
                                Some(copy value_target);
                        }
                    }
                    match copy target_import_resolution.type_target {
                        None => {
                            // Continue.
                        }
                        Some(type_target) => {
                            dest_import_resolution.type_target =
                                Some(copy type_target);
                        }
                    }
                }
            }
        }

        // Add all children from the containing module.
        for containing_module.children.each |ident, name_bindings| {
            if !self.name_is_exported(containing_module, ident) {
                debug!("(resolving glob import) name `%s` is unexported",
                       self.session.str_of(ident));
                loop;
            }

            let mut dest_import_resolution;
            match module_.import_resolutions.find(ident) {
                None => {
                    // Create a new import resolution from this child.
                    dest_import_resolution = @ImportResolution(privacy,
                                                               span);
                    module_.import_resolutions.insert
                        (ident, dest_import_resolution);
                }
                Some(existing_import_resolution) => {
                    dest_import_resolution = existing_import_resolution;
                }
            }


            debug!("(resolving glob import) writing resolution `%s` in `%s` \
                    to `%s`",
                   self.session.str_of(ident),
                   self.module_to_str(containing_module),
                   self.module_to_str(module_));

            // Merge the child item into the import resolution.
            if (*name_bindings).defined_in_namespace(ValueNS) {
                debug!("(resolving glob import) ... for value target");
                dest_import_resolution.value_target =
                    Some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(TypeNS) {
                debug!("(resolving glob import) ... for type target");
                dest_import_resolution.type_target =
                    Some(Target(containing_module, name_bindings));
            }
        }

        debug!("(resolving glob import) successfully resolved import");
        return Success(());
    }

    fn resolve_module_path_from_root(module_: @Module,
                                     module_path: @DVec<ident>,
                                     index: uint,
                                     xray: XrayFlag,
                                     span: span)
                                  -> ResolveResult<@Module> {

        let mut search_module = module_;
        let mut index = index;
        let module_path_len = (*module_path).len();

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.

        while index < module_path_len {
            let name = (*module_path).get_elt(index);
            match self.resolve_name_in_module(search_module, name, TypeNS,
                                              xray) {
                Failed => {
                    self.session.span_err(span, ~"unresolved name");
                    return Failed;
                }
                Indeterminate => {
                    debug!("(resolving module path for import) module \
                            resolution is indeterminate: %s",
                            self.session.str_of(name));
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
                                                               self.session.
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
                                                       self.session.str_of(
                                                            name)));
                            return Failed;
                        }
                    }
                }
            }

            index += 1u;
        }

        return Success(search_module);
    }

    /**
     * Attempts to resolve the module part of an import directive rooted at
     * the given module.
     */
    fn resolve_module_path_for_import(module_: @Module,
                                      module_path: @DVec<ident>,
                                      xray: XrayFlag,
                                      span: span)
                                   -> ResolveResult<@Module> {

        let module_path_len = (*module_path).len();
        assert module_path_len > 0u;

        debug!("(resolving module path for import) processing `%s` rooted at \
               `%s`",
               self.idents_to_str((*module_path).get()),
               self.module_to_str(module_));

        // The first element of the module path must be in the current scope
        // chain.

        let first_element = (*module_path).get_elt(0u);
        let mut search_module;
        match self.resolve_module_in_lexical_scope(module_, first_element) {
            Failed => {
                self.session.span_err(span, ~"unresolved name");
                return Failed;
            }
            Indeterminate => {
                debug!("(resolving module path for import) indeterminate; \
                        bailing");
                return Indeterminate;
            }
            Success(resulting_module) => {
                search_module = resulting_module;
            }
        }

        return self.resolve_module_path_from_root(search_module,
                                               module_path,
                                               1u,
                                               xray,
                                               span);
    }

    fn resolve_item_in_lexical_scope(module_: @Module,
                                     name: ident,
                                     namespace: Namespace)
                                  -> ResolveResult<Target> {

        debug!("(resolving item in lexical scope) resolving `%s` in \
                namespace %? in `%s`",
               self.session.str_of(name),
               namespace,
               self.module_to_str(module_));

        // The current module node is handled specially. First, check for
        // its immediate children.

        match module_.children.find(name) {
            Some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) => {
                return Success(Target(module_, name_bindings));
            }
            Some(_) | None => { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.

        match module_.import_resolutions.find(name) {
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
                        import_resolution.used = true;
                        return Success(copy target);
                    }
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
                ModuleParentLink(parent_module_node, _) |
                BlockParentLink(parent_module_node, _) => {
                    search_module = parent_module_node;
                }
            }

            // Resolve the name in the parent module.
            match self.resolve_name_in_module(search_module, name, namespace,
                                            Xray) {
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

    fn resolve_module_in_lexical_scope(module_: @Module, name: ident)
                                    -> ResolveResult<@Module> {

        match self.resolve_item_in_lexical_scope(module_, name, TypeNS) {
            Success(target) => {
                match target.bindings.type_def {
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

    fn name_is_exported(module_: @Module, name: ident) -> bool {
        return !module_.legacy_exports ||
            module_.exported_names.size() == 0u ||
            module_.exported_names.contains_key(name);
    }

    /**
     * Attempts to resolve the supplied name in the given module for the
     * given namespace. If successful, returns the target corresponding to
     * the name.
     */
    fn resolve_name_in_module(module_: @Module,
                              name: ident,
                              namespace: Namespace,
                              xray: XrayFlag)
                           -> ResolveResult<Target> {

        debug!("(resolving name in module) resolving `%s` in `%s`",
               self.session.str_of(name),
               self.module_to_str(module_));

        if xray == NoXray && !self.name_is_exported(module_, name) {
            debug!("(resolving name in module) name `%s` is unexported",
                   self.session.str_of(name));
            return Failed;
        }

        // First, check the direct children of the module.
        match module_.children.find(name) {
            Some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) => {

                debug!("(resolving name in module) found node as child");
                return Success(Target(module_, name_bindings));
            }
            Some(_) | None => {
                // Continue.
            }
        }

        // Next, check the module's imports. If the module has a glob, then
        // we bail out; we don't know its imports yet.

        if module_.glob_count > 0u {
            debug!("(resolving name in module) module has glob; bailing out");
            return Indeterminate;
        }

        // Otherwise, we check the list of resolved imports.
        match module_.import_resolutions.find(name) {
            Some(import_resolution) => {
                if import_resolution.outstanding_references != 0u {
                    debug!("(resolving name in module) import unresolved; \
                            bailing out");
                    return Indeterminate;
                }

                match (*import_resolution).target_for_namespace(namespace) {
                    None => {
                        debug!("(resolving name in module) name found, but \
                                not in namespace %?",
                               namespace);
                    }
                    Some(target) => {
                        debug!("(resolving name in module) resolved to \
                                import");
                        import_resolution.used = true;
                        return Success(copy target);
                    }
                }
            }
            None => {
                // Continue.
            }
        }

        // We're out of luck.
        debug!("(resolving name in module) failed to resolve %s",
               self.session.str_of(name));
        return Failed;
    }

    /**
     * Resolves a one-level renaming import of the kind `import foo = bar;`
     * This needs special handling, as, unlike all of the other imports, it
     * needs to look in the scope chain for modules and non-modules alike.
     */
    fn resolve_one_level_renaming_import(module_: @Module,
                                         import_directive: @ImportDirective)
                                      -> ResolveResult<()> {

        let mut target_name;
        let mut source_name;
        let allowable_namespaces;
        match *import_directive.subclass {
            SingleImport(target, source, namespaces) => {
                target_name = target;
                source_name = source;
                allowable_namespaces = namespaces;
            }
            GlobImport => {
                fail ~"found `import *`, which is invalid";
            }
        }

        debug!("(resolving one-level naming result) resolving import `%s` = \
                `%s` in `%s`",
                self.session.str_of(target_name),
                self.session.str_of(source_name),
                self.module_to_str(module_));

        // Find the matching items in the lexical scope chain for every
        // namespace. If any of them come back indeterminate, this entire
        // import is indeterminate.

        let mut module_result;
        debug!("(resolving one-level naming result) searching for module");
        match self.resolve_item_in_lexical_scope(module_,
                                                 source_name,
                                                 TypeNS) {
            Failed => {
                debug!("(resolving one-level renaming import) didn't find \
                        module result");
                module_result = None;
            }
            Indeterminate => {
                debug!("(resolving one-level renaming import) module result \
                        is indeterminate; bailing");
                return Indeterminate;
            }
            Success(name_bindings) => {
                debug!("(resolving one-level renaming import) module result \
                        found");
                module_result = Some(copy name_bindings);
            }
        }

        let mut value_result;
        let mut type_result;
        if allowable_namespaces == TypeNSOnly {
            value_result = None;
            type_result = None;
        } else {
            debug!("(resolving one-level naming result) searching for value");
            match self.resolve_item_in_lexical_scope(module_,
                                                   source_name,
                                                   ValueNS) {

                Failed => {
                    debug!("(resolving one-level renaming import) didn't \
                            find value result");
                    value_result = None;
                }
                Indeterminate => {
                    debug!("(resolving one-level renaming import) value \
                            result is indeterminate; bailing");
                    return Indeterminate;
                }
                Success(name_bindings) => {
                    debug!("(resolving one-level renaming import) value \
                            result found");
                    value_result = Some(copy name_bindings);
                }
            }

            debug!("(resolving one-level naming result) searching for type");
            match self.resolve_item_in_lexical_scope(module_,
                                                   source_name,
                                                   TypeNS) {

                Failed => {
                    debug!("(resolving one-level renaming import) didn't \
                            find type result");
                    type_result = None;
                }
                Indeterminate => {
                    debug!("(resolving one-level renaming import) type \
                            result is indeterminate; bailing");
                    return Indeterminate;
                }
                Success(name_bindings) => {
                    debug!("(resolving one-level renaming import) type \
                            result found");
                    type_result = Some(copy name_bindings);
                }
            }
        }

        //
        // NB: This one results in effects that may be somewhat surprising. It
        // means that this:
        //
        // mod A {
        //     impl foo for ... { ... }
        //     mod B {
        //         impl foo for ... { ... }
        //         import bar = foo;
        //         ...
        //     }
        // }
        //
        // results in only A::B::foo being aliased to A::B::bar, not A::foo
        // *and* A::B::foo being aliased to A::B::bar.
        //

        // If nothing at all was found, that's an error.
        if is_none(module_result) &&
                is_none(value_result) &&
                is_none(type_result) {

            self.session.span_err(import_directive.span,
                                  ~"unresolved import");
            return Failed;
        }

        // Otherwise, proceed and write in the bindings.
        match module_.import_resolutions.find(target_name) {
            None => {
                fail ~"(resolving one-level renaming import) reduced graph \
                      construction or glob importing should have created the \
                      import resolution name by now";
            }
            Some(import_resolution) => {
                debug!("(resolving one-level renaming import) writing module \
                        result %? for `%s` into `%s`",
                       is_none(module_result),
                       self.session.str_of(target_name),
                       self.module_to_str(module_));

                import_resolution.value_target = value_result;
                import_resolution.type_target = type_result;

                assert import_resolution.outstanding_references >= 1u;
                import_resolution.outstanding_references -= 1u;
            }
        }

        debug!("(resolving one-level renaming import) successfully resolved");
        return Success(());
    }

    fn report_unresolved_imports(module_: @Module) {
        let index = module_.resolved_import_count;
        let import_count = module_.imports.len();
        if index != import_count {
            self.session.span_err(module_.imports.get_elt(index).span,
                                  ~"unresolved import");
        }

        // Descend into children and anonymous children.
        for module_.children.each |_name, child_node| {
            match child_node.get_module_if_available() {
                None => {
                    // Continue.
                }
                Some(child_module) => {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for module_.anonymous_children.each |_name, module_| {
            self.report_unresolved_imports(module_);
        }
    }

    // Export recording
    //
    // This pass simply determines what all "export" keywords refer to and
    // writes the results into the export map.
    //
    // XXX: This pass will be removed once exports change to per-item. Then
    // this operation can simply be performed as part of item (or import)
    // processing.

    fn record_exports() {
        let root_module = (*self.graph_root).get_module();
        self.record_exports_for_module_subtree(root_module);
    }

    fn record_exports_for_module_subtree(module_: @Module) {
        // If this isn't a local crate, then bail out. We don't need to record
        // exports for local crates.

        match module_.def_id {
            Some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
            }
            None => {
                // Record exports for the root module.
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

        for module_.children.each |_ident, child_name_bindings| {
            match child_name_bindings.get_module_if_available() {
                None => {
                    // Nothing to do.
                }
                Some(child_module) => {
                    self.record_exports_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each |_node_id, child_module| {
            self.record_exports_for_module_subtree(child_module);
        }
    }

    fn record_exports_for_module(module_: @Module) {
        let mut exports2 = ~[];

        if module_.legacy_exports {
            self.add_exports_for_legacy_module(&mut exports2, module_);
        } else {
            self.add_exports_for_module(&mut exports2, module_);
        }
        match copy module_.def_id {
            Some(def_id) => {
                self.export_map2.insert(def_id.node, move exports2);
                debug!("(computing exports) writing exports for %d (some)",
                       def_id.node);
            }
            None => {}
        }
    }


    fn add_exports_of_namebindings(exports2: &mut ~[Export2],
                                   ident: ident,
                                   namebindings: @NameBindings,
                                   reexport: bool) {
        for [ TypeNS, ValueNS ].each |ns| {
            match (namebindings.def_for_namespace(*ns),
                   namebindings.privacy_for_namespace(*ns)) {
                (Some(d), Some(Public)) => {
                    debug!("(computing exports) YES: %s '%s' \
                            => %?",
                           if reexport { ~"reexport" } else { ~"export"},
                           self.session.str_of(ident),
                           def_id_of_def(d));
                    exports2.push(Export2 {
                        reexport: reexport,
                        name: self.session.str_of(ident),
                        def_id: def_id_of_def(d)
                    });
                }
                _ => ()
            }
        }
    }

    fn add_exports_for_module(exports2: &mut ~[Export2], module_: @Module) {

        for module_.children.each_ref |ident, namebindings| {
            debug!("(computing exports) maybe export '%s'",
                   self.session.str_of(*ident));
            self.add_exports_of_namebindings(exports2, *ident,
                                             *namebindings, false)
        }

        for module_.import_resolutions.each_ref |ident, importresolution| {
            for [ TypeNS, ValueNS ].each |ns| {
                match importresolution.target_for_namespace(*ns) {
                    Some(target) => {
                        debug!("(computing exports) maybe reexport '%s'",
                               self.session.str_of(*ident));
                        self.add_exports_of_namebindings(exports2,
                                                         *ident,
                                                         target.bindings,
                                                         true)
                    }
                    _ => ()
                }
            }
        }
    }

    fn add_exports_for_legacy_module(exports2: &mut ~[Export2],
                                     module_: @Module) {
        for module_.exported_names.each |name, _exp_node_id| {
            for self.namespaces.each |namespace| {
                match self.resolve_definition_of_name_in_module(module_,
                                                                name,
                                                                *namespace,
                                                                Xray) {
                    NoNameDefinition => {
                        // Nothing to do.
                    }
                    ChildNameDefinition(target_def) => {
                        debug!("(computing exports) legacy export '%s' \
                                for %?",
                               self.session.str_of(name),
                               module_.def_id);
                        exports2.push(Export2 {
                            reexport: false,
                            name: self.session.str_of(name),
                            def_id: def_id_of_def(target_def)
                        });
                    }
                    ImportNameDefinition(target_def) => {
                        debug!("(computing exports) legacy reexport '%s' for \
                                %?",
                               self.session.str_of(name),
                               module_.def_id);
                        exports2.push(Export2 {
                            reexport: true,
                            name: self.session.str_of(name),
                            def_id: def_id_of_def(target_def)
                        });
                    }
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

    fn with_scope(name: Option<ident>, f: fn()) {
        let orig_module = self.current_module;

        // Move down in the graph.
        match name {
            None => {
                // Nothing to do.
            }
            Some(name) => {
                match orig_module.children.find(name) {
                    None => {
                        debug!("!!! (with scope) didn't find `%s` in `%s`",
                               self.session.str_of(name),
                               self.module_to_str(orig_module));
                    }
                    Some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            None => {
                                debug!("!!! (with scope) didn't find module \
                                        for `%s` in `%s`",
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

        f();

        self.current_module = orig_module;
    }

    // Wraps the given definition in the appropriate number of `def_upvar`
    // wrappers.

    fn upvarify(ribs: @DVec<@Rib>, rib_index: uint, def_like: def_like,
                span: span, allow_capturing_self: AllowCapturingSelfFlag)
             -> Option<def_like> {

        let mut def;
        let mut is_ty_param;

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

        let mut rib_index = rib_index + 1u;
        while rib_index < (*ribs).len() {
            let rib = (*ribs).get_elt(rib_index);
            match rib.kind {
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
                    def_ty_param(did, _) if self.def_map.find(copy(did.node))
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

    fn search_ribs(ribs: @DVec<@Rib>, name: ident, span: span,
                   allow_capturing_self: AllowCapturingSelfFlag)
                -> Option<def_like> {

        // XXX: This should not use a while loop.
        // XXX: Try caching?

        let mut i = (*ribs).len();
        while i != 0 {
            i -= 1;
            let rib = (*ribs).get_elt(i);
            match rib.bindings.find(name) {
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

    fn resolve_crate(@self) {
        debug!("(resolving crate) starting");

        visit_crate(*self.crate, (), mk_vt(@{
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

    fn resolve_item(item: @item, visitor: ResolveVisitor) {
        debug!("(resolving item) resolving %s",
               self.session.str_of(item.ident));

        // Items with the !resolve_unexported attribute are X-ray contexts.
        // This is used to allow the test runner to run unexported tests.
        let orig_xray_flag = self.xray_context;
        if contains_name(attr_metas(item.attrs), ~"!resolve_unexported") {
            self.xray_context = Xray;
        }

        match item.node {

            // enum item: resolve all the variants' discrs,
            // then resolve the ty params
            item_enum(enum_def, type_parameters) => {

                for enum_def.variants.each() |variant| {
                    do variant.node.disr_expr.iter() |dis_expr| {
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
                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, item.id, 0,
                                           NormalRibKind))
                        || {

                    visit_item(item, (), visitor);
                }
            }

            item_ty(_, type_parameters) => {
                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, item.id, 0u,
                                           NormalRibKind))
                        || {

                    visit_item(item, (), visitor);
                }
            }

            item_impl(type_parameters, implemented_traits, self_type,
                      methods) => {

                self.resolve_implementation(item.id, item.span,
                                            type_parameters,
                                            implemented_traits,
                                            self_type, methods, visitor);
            }

            item_trait(type_parameters, traits, methods) => {
                // Create a new rib for the self type.
                let self_type_rib = @Rib(NormalRibKind);
                (*self.type_ribs).push(self_type_rib);
                self_type_rib.bindings.insert(self.self_ident,
                                              dl_def(def_self(item.id)));

                // Create a new rib for the trait-wide type parameters.
                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, item.id, 0u,
                                           NormalRibKind)) {

                    self.resolve_type_parameters(type_parameters, visitor);

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

                    for methods.each |method| {
                        // Create a new rib for the method-specific type
                        // parameters.
                        //
                        // XXX: Do we need a node ID here?

                        match *method {
                          required(ty_m) => {
                            do self.with_type_parameter_rib
                                (HasTypeParameters(&ty_m.tps,
                                                   item.id,
                                                   type_parameters.len(),
                                        MethodRibKind(item.id, Required))) {

                                // Resolve the method-specific type
                                // parameters.
                                self.resolve_type_parameters(ty_m.tps,
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
                                                  type_parameters.len(),
                                                  visitor)
                          }
                        }
                    }
                }

                (*self.type_ribs).pop();
            }

            item_class(struct_def, ty_params) => {
                self.resolve_class(item.id,
                                   @copy ty_params,
                                   struct_def.traits,
                                   struct_def.fields,
                                   struct_def.methods,
                                   struct_def.dtor,
                                   visitor);
            }

            item_mod(module_) => {
                do self.with_scope(Some(item.ident)) {
                    self.resolve_module(module_, item.span, item.ident,
                                        item.id, visitor);
                }
            }

            item_foreign_mod(foreign_module) => {
                do self.with_scope(Some(item.ident)) {
                    for foreign_module.items.each |foreign_item| {
                        match foreign_item.node {
                            foreign_item_fn(_, _, type_parameters) => {
                                do self.with_type_parameter_rib
                                    (HasTypeParameters(&type_parameters,
                                                       foreign_item.id,
                                                       0u,
                                                       OpaqueFunctionRibKind))
                                        || {

                                    visit_foreign_item(*foreign_item, (),
                                                       visitor);
                                }
                            }
                            foreign_item_const(_) => {
                                visit_foreign_item(*foreign_item, (),
                                                   visitor);
                            }
                        }
                    }
                }
            }

            item_fn(fn_decl, _, ty_params, block) => {
                // If this is the main function, we must record it in the
                // session.
                //
                // For speed, we put the string comparison last in this chain
                // of conditionals.

                if !self.session.building_library &&
                    is_none(self.session.main_fn) &&
                    item.ident == syntax::parse::token::special_idents::main {

                    self.session.main_fn = Some((item.id, item.span));
                }

                self.resolve_function(OpaqueFunctionRibKind,
                                      Some(@fn_decl),
                                      HasTypeParameters
                                        (&ty_params,
                                         item.id,
                                         0u,
                                         OpaqueFunctionRibKind),
                                      block,
                                      NoSelfBinding,
                                      NoCaptureClause,
                                      visitor);
            }

            item_const(*) => {
                self.with_constant_rib(|| {
                    visit_item(item, (), visitor);
                });
            }

          item_mac(*) => {
            fail ~"item macros unimplemented"
          }
        }

        self.xray_context = orig_xray_flag;
    }

    fn with_type_parameter_rib(type_parameters: TypeParameters, f: fn()) {
        match type_parameters {
            HasTypeParameters(type_parameters, node_id, initial_index,
                              rib_kind) => {

                let function_type_rib = @Rib(rib_kind);
                (*self.type_ribs).push(function_type_rib);

                for (*type_parameters).eachi |index, type_parameter| {
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
                    (*function_type_rib).bindings.insert(name, def_like);
                }
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }

        f();

        match type_parameters {
            HasTypeParameters(*) => {
                (*self.type_ribs).pop();
            }

            NoTypeParameters => {
                // Nothing to do.
            }
        }
    }

    fn with_label_rib(f: fn()) {
        (*self.label_ribs).push(@Rib(NormalRibKind));
        f();
        (*self.label_ribs).pop();
    }
    fn with_constant_rib(f: fn()) {
        (*self.value_ribs).push(@Rib(ConstantItemRibKind));
        f();
        (*self.value_ribs).pop();
    }


    fn resolve_function(rib_kind: RibKind,
                        optional_declaration: Option<@fn_decl>,
                        type_parameters: TypeParameters,
                        block: blk,
                        self_binding: SelfBinding,
                        capture_clause: CaptureClause,
                        visitor: ResolveVisitor) {
        // Check each element of the capture clause.
        match capture_clause {
            NoCaptureClause => {
                // Nothing to do.
            }
            HasCaptureClause(capture_clause) => {
                // Resolve each captured item.
                for (*capture_clause).each |capture_item| {
                    match self.resolve_identifier(capture_item.name,
                                                  ValueNS,
                                                  true,
                                                  capture_item.span) {
                        None => {
                            self.session.span_err(capture_item.span,
                                                  ~"unresolved name in \
                                                   capture clause");
                        }
                        Some(def) => {
                            self.record_def(capture_item.id, def);
                        }
                    }
                }
            }
        }

        // Create a value rib for the function.
        let function_value_rib = @Rib(rib_kind);
        (*self.value_ribs).push(function_value_rib);

        // Create a label rib for the function.
        let function_label_rib = @Rib(rib_kind);
        (*self.label_ribs).push(function_label_rib);

        // If this function has type parameters, add them now.
        do self.with_type_parameter_rib(type_parameters) {
            // Resolve the type parameters.
            match type_parameters {
                NoTypeParameters => {
                    // Continue.
                }
                HasTypeParameters(type_parameters, _, _, _) => {
                    self.resolve_type_parameters(*type_parameters, visitor);
                }
            }

            // Add self to the rib, if necessary.
            match self_binding {
                NoSelfBinding => {
                    // Nothing to do.
                }
                HasSelfBinding(self_node_id) => {
                    let def_like = dl_def(def_self(self_node_id));
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
                        let name = argument.ident;
                        let def_like = dl_def(def_arg(argument.id,
                                                      argument.mode));
                        (*function_value_rib).bindings.insert(name, def_like);

                        self.resolve_type(argument.ty, visitor);

                        debug!("(resolving function) recorded argument `%s`",
                               self.session.str_of(name));
                    }

                    self.resolve_type(declaration.output, visitor);
                }
            }

            // Resolve the function body.
            self.resolve_block(block, visitor);

            debug!("(resolving function) leaving function");
        }

        (*self.label_ribs).pop();
        (*self.value_ribs).pop();
    }

    fn resolve_type_parameters(type_parameters: ~[ty_param],
                               visitor: ResolveVisitor) {
        for type_parameters.each |type_parameter| {
            for type_parameter.bounds.each |bound| {
                match *bound {
                    bound_copy | bound_send | bound_const | bound_owned => {
                        // Nothing to do.
                    }
                    bound_trait(trait_type) => {
                        self.resolve_type(trait_type, visitor);
                    }
                }
            }
        }
    }

    fn resolve_class(id: node_id,
                     type_parameters: @~[ty_param],
                     traits: ~[@trait_ref],
                     fields: ~[@struct_field],
                     methods: ~[@method],
                     optional_destructor: Option<class_dtor>,
                     visitor: ResolveVisitor) {
        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = (*type_parameters).len();
        let borrowed_type_parameters: &~[ty_param] = &*type_parameters;
        do self.with_type_parameter_rib(HasTypeParameters
                                        (borrowed_type_parameters, id, 0,
                                         OpaqueFunctionRibKind)) {

            // Resolve the type parameters.
            self.resolve_type_parameters(*type_parameters, visitor);

            // Resolve implemented traits.
            for traits.each |trt| {
                match self.resolve_path(trt.path, TypeNS, true, visitor) {
                    None => {
                        self.session.span_err(trt.path.span,
                                              ~"attempt to implement a \
                                               nonexistent trait");
                    }
                    Some(def) => {
                        // Write a mapping from the trait ID to the
                        // definition of the trait into the definition
                        // map.

                        debug!("(resolving class) found trait def: %?", def);

                        self.record_def(trt.ref_id, def);

                        // XXX: This is wrong but is needed for tests to
                        // pass.

                        self.record_def(id, def);
                    }
                }
            }

            // Resolve methods.
            for methods.each |method| {
                self.resolve_method(MethodRibKind(id, Provided(method.id)),
                                    *method,
                                    outer_type_parameter_count,
                                    visitor);
            }

            // Resolve fields.
            for fields.each |field| {
                self.resolve_type(field.node.ty, visitor);
            }

            // Resolve the destructor, if applicable.
            match optional_destructor {
                None => {
                    // Nothing to do.
                }
                Some(destructor) => {
                    self.resolve_function(NormalRibKind,
                                          None,
                                          NoTypeParameters,
                                          destructor.node.body,
                                          HasSelfBinding
                                            (destructor.node.self_id),
                                          NoCaptureClause,
                                          visitor);
                }
            }
        }
    }

    // Does this really need to take a RibKind or is it always going
    // to be NormalRibKind?
    fn resolve_method(rib_kind: RibKind,
                      method: @method,
                      outer_type_parameter_count: uint,
                      visitor: ResolveVisitor) {
        let borrowed_method_type_parameters = &method.tps;
        let type_parameters =
            HasTypeParameters(borrowed_method_type_parameters,
                              method.id,
                              outer_type_parameter_count,
                              rib_kind);
        // we only have self ty if it is a non static method
        let self_binding = match method.self_ty.node {
          sty_static => { NoSelfBinding }
          _ => { HasSelfBinding(method.self_id) }
        };

        self.resolve_function(rib_kind,
                              Some(@method.decl),
                              type_parameters,
                              method.body,
                              self_binding,
                              NoCaptureClause,
                              visitor);
    }

    fn resolve_implementation(id: node_id,
                              span: span,
                              type_parameters: ~[ty_param],
                              opt_trait_reference: Option<@trait_ref>,
                              self_type: @Ty,
                              methods: ~[@method],
                              visitor: ResolveVisitor) {
        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = type_parameters.len();
        let borrowed_type_parameters: &~[ty_param] = &type_parameters;
        do self.with_type_parameter_rib(HasTypeParameters
                                        (borrowed_type_parameters, id, 0u,
                                         NormalRibKind)) {
            // Resolve the type parameters.
            self.resolve_type_parameters(type_parameters, visitor);

            // Resolve the trait reference, if necessary.
            let original_trait_refs = self.current_trait_refs;
            match opt_trait_reference {
                Some(trait_reference) => {
                    let new_trait_refs = @DVec();
                    match self.resolve_path(
                        trait_reference.path, TypeNS, true, visitor) {
                        None => {
                            self.session.span_err(span,
                                                  ~"attempt to implement an \
                                                    unknown trait");
                        }
                        Some(def) => {
                            self.record_def(trait_reference.ref_id, def);

                            // Record the current trait reference.
                            (*new_trait_refs).push(def_id_of_def(def));
                        }
                    }
                    // Record the current set of trait references.
                    self.current_trait_refs = Some(new_trait_refs);
                }
                None => ()
            }

            // Resolve the self type.
            self.resolve_type(self_type, visitor);

            for methods.each |method| {
                // We also need a new scope for the method-specific
                // type parameters.
                self.resolve_method(MethodRibKind(id, Provided(method.id)),
                                    *method,
                                    outer_type_parameter_count,
                                    visitor);
/*
                let borrowed_type_parameters = &method.tps;
                self.resolve_function(MethodRibKind(id, Provided(method.id)),
                                      Some(@method.decl),
                                      HasTypeParameters
                                        (borrowed_type_parameters,
                                         method.id,
                                         outer_type_parameter_count,
                                         NormalRibKind),
                                      method.body,
                                      HasSelfBinding(method.self_id),
                                      NoCaptureClause,
                                      visitor);
*/
            }

            // Restore the original trait references.
            self.current_trait_refs = original_trait_refs;
        }
    }

    fn resolve_module(module_: _mod, span: span, _name: ident, id: node_id,
                      visitor: ResolveVisitor) {

        // Write the implementations in scope into the module metadata.
        debug!("(resolving module) resolving module ID %d", id);
        visit_mod(module_, span, id, (), visitor);
    }

    fn resolve_local(local: @local, visitor: ResolveVisitor) {
        let mut mutability;
        if local.node.is_mutbl {
            mutability = Mutable;
        } else {
            mutability = Immutable;
        }

        // Resolve the type.
        self.resolve_type(local.node.ty, visitor);

        // Resolve the initializer, if necessary.
        match local.node.init {
            None => {
                // Nothing to do.
            }
            Some(initializer) => {
                self.resolve_expr(initializer.expr, visitor);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern(local.node.pat, IrrefutableMode, mutability,
                             None, visitor);
    }

    fn binding_mode_map(pat: @pat) -> BindingMap {
        let result = HashMap();
        do pat_bindings(self.def_map, pat) |binding_mode, _id, sp, path| {
            let ident = path_to_ident(path);
            result.insert(ident,
                          binding_info {span: sp,
                                        binding_mode: binding_mode});
        }
        return result;
    }

    fn check_consistent_bindings(arm: arm) {
        if arm.pats.len() == 0 { return; }
        let map_0 = self.binding_mode_map(arm.pats[0]);
        for arm.pats.eachi() |i, p| {
            let map_i = self.binding_mode_map(*p);

            for map_0.each |key, binding_0| {
                match map_i.find(key) {
                  None => {
                    self.session.span_err(
                        p.span,
                        fmt!("variable `%s` from pattern #1 is \
                                  not bound in pattern #%u",
                             self.session.str_of(key), i + 1));
                  }
                  Some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.session.span_err(
                            binding_i.span,
                            fmt!("variable `%s` is bound with different \
                                      mode in pattern #%u than in pattern #1",
                                 self.session.str_of(key), i + 1));
                    }
                  }
                }
            }

            for map_i.each |key, binding| {
                if !map_0.contains_key(key) {
                    self.session.span_err(
                        binding.span,
                        fmt!("variable `%s` from pattern #%u is \
                                  not bound in pattern #1",
                             self.session.str_of(key), i + 1));
                }
            }
        }
    }

    fn resolve_arm(arm: arm, visitor: ResolveVisitor) {
        (*self.value_ribs).push(@Rib(NormalRibKind));

        let bindings_list = HashMap();
        for arm.pats.each |pattern| {
            self.resolve_pattern(*pattern, RefutableMode, Immutable,
                                 Some(bindings_list), visitor);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        visit_expr_opt(arm.guard, (), visitor);
        self.resolve_block(arm.body, visitor);

        (*self.value_ribs).pop();
    }

    fn resolve_block(block: blk, visitor: ResolveVisitor) {
        debug!("(resolving block) entering block");
        (*self.value_ribs).push(@Rib(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        match self.current_module.anonymous_children.find(block.node.id) {
            None => { /* Nothing to do. */ }
            Some(anonymous_module) => {
                debug!("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module;
            }
        }

        // Descend into the block.
        visit_block(block, (), visitor);

        // Move back up.
        self.current_module = orig_module;

        (*self.value_ribs).pop();
        debug!("(resolving block) leaving block");
    }

    fn resolve_type(ty: @Ty, visitor: ResolveVisitor) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            ty_path(path, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // scopes looking for it.
                let mut result_def = None;

                // First, check to see whether the name is a primitive type.
                if path.idents.len() == 1u {
                    let name = path.idents.last();

                    match self.primitive_type_table
                            .primitive_types
                            .find(name) {

                        Some(primitive_type) => {
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
                                        type",
                                       self.session.str_of(
                                            path.idents.last()));
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

                match copy result_def {
                    Some(def) => {
                        // Write the result into the def map.
                        debug!("(resolving type) writing resolution for `%s` \
                                (id %d)",
                               connect(path.idents.map(
                                   |x| self.session.str_of(*x)), ~"::"),
                               path_id);
                        self.record_def(path_id, def);
                    }
                    None => {
                        self.session.span_err
                            (ty.span, fmt!("use of undeclared type name `%s`",
                                           connect(path.idents.map(
                                               |x| self.session.str_of(*x)),
                                                   ~"::")));
                    }
                }
            }

            _ => {
                // Just resolve embedded types.
                visit_ty(ty, (), visitor);
            }
        }
    }

    fn resolve_pattern(pattern: @pat,
                       mode: PatternBindingMode,
                       mutability: Mutability,
                       // Maps idents to the node ID for the (outermost)
                       // pattern that binds them
                       bindings_list: Option<HashMap<ident,node_id>>,
                       visitor: ResolveVisitor) {

        let pat_id = pattern.id;
        do walk_pat(pattern) |pattern| {
            match pattern.node {
                pat_ident(binding_mode, path, _)
                        if !path.global && path.idents.len() == 1u => {

                    // The meaning of pat_ident with no type parameters
                    // depends on whether an enum variant with that name is in
                    // scope. The probing lookup has to be careful not to emit
                    // spurious errors. Only matching patterns (match) can
                    // match nullary variants. For binding patterns (let),
                    // matching such a variant is simply disallowed (since
                    // it's rarely what you want).

                    let ident = path.idents[0];

                    match self.resolve_enum_variant_or_const(ident) {
                        FoundEnumVariant(def) if mode == RefutableMode => {
                            debug!("(resolving pattern) resolving `%s` to \
                                    enum variant",
                                    self.session.str_of(ident));

                            self.record_def(pattern.id, def);
                        }
                        FoundEnumVariant(_) => {
                            self.session.span_err(pattern.span,
                                                  fmt!("declaration of `%s` \
                                                        shadows an enum \
                                                        that's in scope",
                                                        self.session
                                                        .str_of(ident)));
                        }
                        FoundConst => {
                            self.session.span_err(pattern.span,
                                                  ~"pattern variable \
                                                   conflicts with a constant \
                                                   in scope");
                        }
                        EnumVariantOrConstNotFound => {
                            debug!("(resolving pattern) binding `%s`",
                                   self.session.str_of(ident));

                            let is_mutable = mutability == Mutable;

                            let def = match mode {
                                RefutableMode => {
                                    // For pattern arms, we must use
                                    // `def_binding` definitions.

                                    def_binding(pattern.id, binding_mode)
                                }
                                IrrefutableMode => {
                                    // But for locals, we use `def_local`.
                                    def_local(pattern.id, is_mutable)
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
                                if !bindings_list.contains_key(ident) => {
                                    let last_rib = (*self.value_ribs).last();
                                    last_rib.bindings.insert(ident,
                                                             dl_def(def));
                                    bindings_list.insert(ident, pat_id);
                                }
                                Some(b) => {
                                  if b.find(ident) == Some(pat_id) {
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
                                    let last_rib = (*self.value_ribs).last();
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

                pat_ident(_, path, _) | pat_enum(path, _) => {
                    // These two must be enum variants.
                    match self.resolve_path(path, ValueNS, false, visitor) {
                        Some(def @ def_variant(*)) => {
                            self.record_def(pattern.id, def);
                        }
                        Some(_) => {
                            self.session.span_err(
                                path.span,
                                fmt!("not an enum variant: %s",
                                     self.session.str_of(
                                         path.idents.last())));
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

                pat_lit(expr) => {
                    self.resolve_expr(expr, visitor);
                }

                pat_range(first_expr, last_expr) => {
                    self.resolve_expr(first_expr, visitor);
                    self.resolve_expr(last_expr, visitor);
                }

                pat_struct(path, _, _) => {
                    match self.resolve_path(path, TypeNS, false, visitor) {
                        Some(def_ty(class_id))
                                if self.structs.contains_key(class_id) => {
                            let class_def = def_class(class_id);
                            self.record_def(pattern.id, class_def);
                        }
                        Some(definition @ def_variant(_, variant_id))
                                if self.structs.contains_key(variant_id) => {
                            self.record_def(pattern.id, definition);
                        }
                        _ => {
                            self.session.span_err(
                                path.span,
                                fmt!("`%s` does not name a structure",
                                     connect(path.idents.map(
                                         |x| self.session.str_of(*x)),
                                             ~"::")));
                        }
                    }
                }

                _ => {
                    // Nothing to do.
                }
            }
        }
    }

    fn resolve_enum_variant_or_const(name: ident)
                                  -> EnumVariantOrConstResolution {

        match self.resolve_item_in_lexical_scope(self.current_module,
                                               name,
                                               ValueNS) {

            Success(target) => {
                match target.bindings.value_def {
                    None => {
                        fail ~"resolved name in the value namespace to a set \
                              of name bindings with no def?!";
                    }
                    Some(def) => {
                        match def.def {
                            def @ def_variant(*) => {
                                return FoundEnumVariant(def);
                            }
                            def_const(*) => {
                                return FoundConst;
                            }
                            _ => {
                                return EnumVariantOrConstNotFound;
                            }
                        }
                    }
                }
            }

            Indeterminate => {
                fail ~"unexpected indeterminate result";
            }

            Failed => {
                return EnumVariantOrConstNotFound;
            }
        }
    }

    /**
     * If `check_ribs` is true, checks the local definitions first; i.e.
     * doesn't skip straight to the containing module.
     */
    fn resolve_path(path: @path, namespace: Namespace, check_ribs: bool,
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

        return self.resolve_identifier(path.idents.last(),
                                       namespace,
                                       check_ribs,
                                       path.span);
    }

    fn resolve_identifier(identifier: ident,
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

    // XXX: Merge me with resolve_name_in_module?
    fn resolve_definition_of_name_in_module(containing_module: @Module,
                                            name: ident,
                                            namespace: Namespace,
                                            xray: XrayFlag)
                                         -> NameDefinition {

        if xray == NoXray && !self.name_is_exported(containing_module, name) {
            debug!("(resolving definition of name in module) name `%s` is \
                    unexported",
                   self.session.str_of(name));
            return NoNameDefinition;
        }

        // First, search children.
        match containing_module.children.find(name) {
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
        match containing_module.import_resolutions.find(name) {
            Some(import_resolution) if import_resolution.privacy == Public ||
                                       xray == Xray => {
                match (*import_resolution).target_for_namespace(namespace) {
                    Some(target) => {
                        match (target.bindings.def_for_namespace(namespace),
                               target.bindings.privacy_for_namespace(
                                    namespace)) {
                            (Some(def), Some(Public)) => {
                                // Found it.
                                import_resolution.used = true;
                                return ImportNameDefinition(def);
                            }
                            (Some(_), _) | (None, _) => {
                                // This can happen with external impls, due to
                                // the imperfect way we read the metadata.

                                return NoNameDefinition;
                            }
                        }
                    }
                    None => {
                        return NoNameDefinition;
                    }
                }
            }
            Some(_) | None => {
                return NoNameDefinition;
            }
        }
    }

    fn intern_module_part_of_path(path: @path) -> @DVec<ident> {
        let module_path_idents = @DVec();
        for path.idents.eachi |index, ident| {
            if index == path.idents.len() - 1u {
                break;
            }

            (*module_path_idents).push(*ident);
        }

        return module_path_idents;
    }

    fn resolve_module_relative_path(path: @path,
                                    +xray: XrayFlag,
                                    namespace: Namespace)
                                 -> Option<def> {

        let module_path_idents = self.intern_module_part_of_path(path);

        let mut containing_module;
        match self.resolve_module_path_for_import(self.current_module,
                                                module_path_idents,
                                                xray,
                                                path.span) {

            Failed => {
                self.session.span_err(path.span,
                                      fmt!("use of undeclared module `%s`",
                                           self.idents_to_str(
                                               (*module_path_idents).get())));
                return None;
            }

            Indeterminate => {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = path.idents.last();
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

    fn resolve_crate_relative_path(path: @path,
                                   +xray: XrayFlag,
                                   namespace: Namespace)
                                -> Option<def> {

        let module_path_idents = self.intern_module_part_of_path(path);

        let root_module = (*self.graph_root).get_module();

        let mut containing_module;
        match self.resolve_module_path_from_root(root_module,
                                               module_path_idents,
                                               0u,
                                               xray,
                                               path.span) {

            Failed => {
                self.session.span_err(path.span,
                                      fmt!("use of undeclared module `::%s`",
                                            self.idents_to_str
                                              ((*module_path_idents).get())));
                return None;
            }

            Indeterminate => {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = path.idents.last();
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

    fn resolve_identifier_in_local_ribs(ident: ident,
                                        namespace: Namespace,
                                        span: span)
                                     -> Option<def> {
        // Check the local set of ribs.
        let mut search_result;
        match namespace {
            ValueNS => {
                search_result = self.search_ribs(self.value_ribs, ident, span,
                                                 DontAllowCapturingSelf);
            }
            TypeNS => {
                search_result = self.search_ribs(self.type_ribs, ident, span,
                                                 AllowCapturingSelf);
            }
        }

        match copy search_result {
            Some(dl_def(def)) => {
                debug!("(resolving path in local ribs) resolved `%s` to \
                        local: %?",
                       self.session.str_of(ident),
                       def);
                return Some(def);
            }
            Some(dl_field) | Some(dl_impl(_)) | None => {
                return None;
            }
        }
    }

    fn resolve_item_by_identifier_in_lexical_scope(ident: ident,
                                                   namespace: Namespace)
                                                -> Option<def> {
        // Check the items.
        match self.resolve_item_in_lexical_scope(self.current_module,
                                               ident,
                                               namespace) {
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
                               self.session.str_of(ident));
                        return Some(def);
                    }
                }
            }
            Indeterminate => {
                fail ~"unexpected indeterminate result";
            }
            Failed => {
                return None;
            }
        }
    }

    fn name_exists_in_scope_class(name: &str) -> bool {
        let mut i = self.type_ribs.len();
        while i != 0 {
          i -= 1;
          let rib = self.type_ribs.get_elt(i);
          match rib.kind {
            MethodRibKind(node_id, _) =>
              for vec::each(self.crate.node.module.items) |item| {
                if item.id == node_id {
                  match item.node {
                    item_class(class_def, _) => {
                      for vec::each(class_def.fields) |field| {
                        match field.node.kind {
                          syntax::ast::unnamed_field
                            => {},
                          syntax::ast::named_field(ident, _, _)
                            => {
                              if str::eq_slice(self.session.str_of(ident),
                                               name) {
                                return true
                              }
                            }
                        }
                      }
                      for vec::each(class_def.methods) |method| {
                        if str::eq_slice(self.session.str_of(method.ident),
                                         name) {
                          return true
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

    fn resolve_expr(expr: @expr, visitor: ResolveVisitor) {
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
                               connect(path.idents.map(
                                   |x| self.session.str_of(*x)), ~"::"));
                        self.record_def(expr.id, def);
                    }
                    None => {
                        let wrong_name =
                            connect(path.idents.map(
                                |x| self.session.str_of(*x)), ~"::") ;
                        if self.name_exists_in_scope_class(wrong_name) {
                            self.session.span_err(expr.span,
                                        fmt!("unresolved name: `%s`. \
                                            Did you mean: `self.%s`?",
                                        wrong_name,
                                        wrong_name));
                        }
                        else {
                            self.session.span_err(expr.span,
                                                fmt!("unresolved name: %s",
                                                wrong_name));
                        }
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_fn(_, fn_decl, block, capture_clause) |
            expr_fn_block(fn_decl, block, capture_clause) => {
                self.resolve_function(FunctionRibKind(expr.id, block.node.id),
                                      Some(@fn_decl),
                                      NoTypeParameters,
                                      block,
                                      NoSelfBinding,
                                      HasCaptureClause(capture_clause),
                                      visitor);
            }

            expr_struct(path, _, _) => {
                // Resolve the path to the structure it goes to.
                //
                // XXX: We might want to support explicit type parameters in
                // the path, in which case this gets a little more
                // complicated:
                //
                // 1. Should we go through the ast_path_to_ty() path, which
                //    handles typedefs and the like?
                //
                // 2. If so, should programmers be able to write this?
                //
                //    class Foo<A> { ... }
                //    type Bar<A> = Foo<A>;
                //    let bar = Bar { ... } // no type parameters

                match self.resolve_path(path, TypeNS, false, visitor) {
                    Some(def_ty(class_id)) | Some(def_class(class_id))
                            if self.structs.contains_key(class_id) => {
                        let class_def = def_class(class_id);
                        self.record_def(expr.id, class_def);
                    }
                    Some(definition @ def_variant(_, class_id))
                            if self.structs.contains_key(class_id) => {
                        self.record_def(expr.id, definition);
                    }
                    _ => {
                        self.session.span_err(
                            path.span,
                            fmt!("`%s` does not name a structure",
                                 connect(path.idents.map(
                                     |x| self.session.str_of(*x)),
                                         ~"::")));
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_loop(_, Some(label)) => {
                do self.with_label_rib {
                    let def_like = dl_def(def_label(expr.id));
                    self.label_ribs.last().bindings.insert(label, def_like);

                    visit_expr(expr, (), visitor);
                }
            }

            expr_break(Some(label)) | expr_again(Some(label)) => {
                match self.search_ribs(self.label_ribs, label, expr.span,
                                       DontAllowCapturingSelf) {
                    None =>
                        self.session.span_err(expr.span,
                                              fmt!("use of undeclared label \
                                                   `%s`", self.session.str_of(
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

    fn record_candidate_traits_for_expr_if_necessary(expr: @expr) {
        match expr.node {
            expr_field(_, ident, _) => {
                let traits = self.search_for_traits_containing_method(ident);
                self.trait_map.insert(expr.id, traits);
            }
            expr_binary(add, _, _) | expr_assign_op(add, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.add_trait);
            }
            expr_binary(subtract, _, _) | expr_assign_op(subtract, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.sub_trait);
            }
            expr_binary(mul, _, _) | expr_assign_op(mul, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.mul_trait);
            }
            expr_binary(div, _, _) | expr_assign_op(div, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.div_trait);
            }
            expr_binary(rem, _, _) | expr_assign_op(rem, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.modulo_trait);
            }
            expr_binary(bitxor, _, _) | expr_assign_op(bitxor, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitxor_trait);
            }
            expr_binary(bitand, _, _) | expr_assign_op(bitand, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitand_trait);
            }
            expr_binary(bitor, _, _) | expr_assign_op(bitor, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.bitor_trait);
            }
            expr_binary(shl, _, _) | expr_assign_op(shl, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.shl_trait);
            }
            expr_binary(shr, _, _) | expr_assign_op(shr, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.shr_trait);
            }
            expr_binary(lt, _, _) | expr_binary(le, _, _) |
            expr_binary(ge, _, _) | expr_binary(gt, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.ord_trait);
            }
            expr_binary(eq, _, _) | expr_binary(ne, _, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.eq_trait);
            }
            expr_unary(neg, _) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.neg_trait);
            }
            expr_index(*) => {
                self.add_fixed_trait_for_expr(expr.id,
                                              self.lang_items.index_trait);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn search_for_traits_containing_method(name: ident) -> @DVec<def_id> {
        debug!("(searching for traits containing method) looking for '%s'",
               self.session.str_of(name));

        let found_traits = @DVec();
        let mut search_module = self.current_module;
        loop {
            // Look for the current trait.
            match copy self.current_trait_refs {
                Some(trait_def_ids) => {
                    for trait_def_ids.each |trait_def_id| {
                        self.add_trait_info_if_containing_method(
                            found_traits, *trait_def_id, name);
                    }
                }
                None => {
                    // Nothing to do.
                }
            }

            // Look for trait children.
            for search_module.children.each |_name, child_name_bindings| {
                match child_name_bindings.def_for_namespace(TypeNS) {
                    Some(def) => {
                        match def {
                            def_ty(trait_def_id) => {
                                self.add_trait_info_if_containing_method(
                                    found_traits, trait_def_id, name);
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
            for search_module.import_resolutions.each
                    |_ident, import_resolution| {

                match import_resolution.target_for_namespace(TypeNS) {
                    None => {
                        // Continue.
                    }
                    Some(target) => {
                        match target.bindings.def_for_namespace(TypeNS) {
                            Some(def) => {
                                match def {
                                    def_ty(trait_def_id) => {
                                        self.
                                        add_trait_info_if_containing_method(
                                        found_traits, trait_def_id, name);
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

    fn add_trait_info_if_containing_method(found_traits: @DVec<def_id>,
                                           trait_def_id: def_id,
                                           name: ident) {

        debug!("(adding trait info if containing method) trying trait %d:%d \
                for method '%s'",
               trait_def_id.crate,
               trait_def_id.node,
               self.session.str_of(name));

        match self.trait_info.find(trait_def_id) {
            Some(trait_info) if trait_info.contains_key(name) => {
                debug!("(adding trait info if containing method) found trait \
                        %d:%d for method '%s'",
                       trait_def_id.crate,
                       trait_def_id.node,
                       self.session.str_of(name));
                (*found_traits).push(trait_def_id);
            }
            Some(_) | None => {
                // Continue.
            }
        }
    }

    fn add_fixed_trait_for_expr(expr_id: node_id, +trait_id: Option<def_id>) {
        let traits = @DVec();
        traits.push(trait_id.get());
        self.trait_map.insert(expr_id, traits);
    }

    fn record_def(node_id: node_id, def: def) {
        debug!("(recording def) recording %? for %?", def, node_id);
        self.def_map.insert(node_id, def);
    }

    //
    // Unused import checking
    //
    // Although this is a lint pass, it lives in here because it depends on
    // resolve data structures.
    //

    fn check_for_unused_imports_if_necessary() {
        if self.unused_import_lint_level == allow {
            return;
        }

        let root_module = (*self.graph_root).get_module();
        self.check_for_unused_imports_in_module_subtree(root_module);
    }

    fn check_for_unused_imports_in_module_subtree(module_: @Module) {
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

        for module_.children.each |_ident, child_name_bindings| {
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

        for module_.anonymous_children.each |_node_id, child_module| {
            self.check_for_unused_imports_in_module_subtree(child_module);
        }
    }

    fn check_for_unused_imports_in_module(module_: @Module) {
        for module_.import_resolutions.each |_name, import_resolution| {
            if !import_resolution.used {
                match self.unused_import_lint_level {
                    warn => {
                        self.session.span_warn(import_resolution.span,
                                               ~"unused import");
                    }
                    deny | forbid => {
                      self.session.span_err(import_resolution.span,
                                            ~"unused import");
                    }
                    allow => {
                      self.session.span_bug(import_resolution.span,
                                            ~"shouldn't be here if lint \
                                              is allowed");
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

    /// A somewhat inefficient routine to print out the name of a module.
    fn module_to_str(module_: @Module) -> ~str {
        let idents = DVec();
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
                    idents.push(syntax::parse::token::special_idents::opaque);
                    current_module = module_;
                }
            }
        }

        if idents.len() == 0u {
            return ~"???";
        }

        let mut string = ~"";
        let mut i = idents.len() - 1u;
        loop {
            if i < idents.len() - 1u {
                string += ~"::";
            }
            string += self.session.str_of(idents.get_elt(i));

            if i == 0u {
                break;
            }
            i -= 1u;
        }

        return string;
    }

    fn dump_module(module_: @Module) {
        debug!("Dump of module `%s`:", self.module_to_str(module_));

        debug!("Children:");
        for module_.children.each |name, _child| {
            debug!("* %s", self.session.str_of(name));
        }

        debug!("Import resolutions:");
        for module_.import_resolutions.each |name, import_resolution| {
            let mut value_repr;
            match (*import_resolution).target_for_namespace(ValueNS) {
                None => { value_repr = ~""; }
                Some(_) => {
                    value_repr = ~" value:?";
                    // XXX
                }
            }

            let mut type_repr;
            match (*import_resolution).target_for_namespace(TypeNS) {
                None => { type_repr = ~""; }
                Some(_) => {
                    type_repr = ~" type:?";
                    // XXX
                }
            }

            debug!("* %s:%s%s", self.session.str_of(name),
                   value_repr, type_repr);
        }
    }
}

/// Entry point to crate resolution.
fn resolve_crate(session: Session, lang_items: LanguageItems, crate: @crate)
              -> { def_map: DefMap,
                   exp_map2: ExportMap2,
                   trait_map: TraitMap } {

    let resolver = @Resolver(session, lang_items, crate);
    resolver.resolve(resolver);
    return {
        def_map: resolver.def_map,
        exp_map2: resolver.export_map2,
        trait_map: resolver.trait_map
    };
}

