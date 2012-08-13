import driver::session::session;
import metadata::csearch::{each_path};
import metadata::csearch::{get_method_names_if_trait, lookup_defs};
import metadata::cstore::find_use_stmt_cnum;
import metadata::decoder::{def_like, dl_def, dl_field, dl_impl};
import middle::lang_items::LanguageItems;
import middle::lint::{deny, allow, forbid, level, unused_imports, warn};
import middle::pat_util::{pat_bindings};
import syntax::ast::{_mod, add, arm};
import syntax::ast::{bind_by_ref, bind_by_implicit_ref, bind_by_value};
import syntax::ast::{bitand, bitor, bitxor};
import syntax::ast::{blk, bound_const, bound_copy, bound_owned, bound_send};
import syntax::ast::{bound_trait, binding_mode,
                     capture_clause, class_ctor, class_dtor};
import syntax::ast::{class_member, class_method, crate, crate_num, decl_item};
import syntax::ast::{def, def_arg, def_binding, def_class, def_const, def_fn};
import syntax::ast::{def_foreign_mod, def_id, def_local, def_mod};
import syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
import syntax::ast::{def_typaram_binder, def_static_method};
import syntax::ast::{def_upvar, def_use, def_variant, expr, expr_assign_op};
import syntax::ast::{expr_binary, expr_cast, expr_field, expr_fn};
import syntax::ast::{expr_fn_block, expr_index, expr_path};
import syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
import syntax::ast::{def_upvar, def_use, def_variant, div, eq};
import syntax::ast::{enum_variant_kind, expr, expr_assign_op, expr_binary};
import syntax::ast::{expr_cast, expr_field, expr_fn, expr_fn_block};
import syntax::ast::{expr_index, expr_path, expr_struct, expr_unary, fn_decl};
import syntax::ast::{foreign_item, foreign_item_fn, ge, gt, ident, trait_ref};
import syntax::ast::{impure_fn, instance_var, item, item_class, item_const};
import syntax::ast::{item_enum, item_fn, item_mac, item_foreign_mod};
import syntax::ast::{item_impl, item_mod, item_trait, item_ty, le, local};
import syntax::ast::{local_crate, lt, method, mul, ne, neg, node_id, pat};
import syntax::ast::{pat_enum, pat_ident, path, prim_ty, pat_box, pat_uniq};
import syntax::ast::{pat_lit, pat_range, pat_rec, pat_struct, pat_tup};
import syntax::ast::{pat_wild, provided, required, rem, self_ty_, shl};
import syntax::ast::{stmt_decl, struct_variant_kind, sty_static, subtract};
import syntax::ast::{tuple_variant_kind, ty};
import syntax::ast::{ty_bool, ty_char, ty_f, ty_f32, ty_f64, ty_float, ty_i};
import syntax::ast::{ty_i16, ty_i32, ty_i64, ty_i8, ty_int, ty_param};
import syntax::ast::{ty_path, ty_str, ty_u, ty_u16, ty_u32, ty_u64, ty_u8};
import syntax::ast::{ty_uint, variant, view_item, view_item_export};
import syntax::ast::{view_item_import, view_item_use, view_path_glob};
import syntax::ast::{view_path_list, view_path_simple};
import syntax::ast_util::{def_id_of_def, dummy_sp, local_def, new_def_hash};
import syntax::ast_util::{path_to_ident, walk_pat, trait_method_to_ty_method};
import syntax::attr::{attr_metas, contains_name};
import syntax::print::pprust::{pat_to_str, path_to_str};
import syntax::codemap::span;
import syntax::visit::{default_visitor, fk_method, mk_vt, visit_block};
import syntax::visit::{visit_crate, visit_expr, visit_expr_opt, visit_fn};
import syntax::visit::{visit_foreign_item, visit_item, visit_method_helper};
import syntax::visit::{visit_mod, visit_ty, vt};

import box::ptr_eq;
import dvec::dvec;
import option::{get, is_some};
import str::{connect, split_str};
import vec::pop;

import std::list::{cons, list, nil};
import std::map::{hashmap, int_hash, box_str_hash};
import str_eq = str::eq;

// Definition mapping
type DefMap = hashmap<node_id,def>;

struct binding_info {
    span: span;
    binding_mode: binding_mode;
}

// Map from the name in a pattern to its binding mode.
type BindingMap = hashmap<ident,binding_info>;

// Implementation resolution

// XXX: This kind of duplicates information kept in ty::method. Maybe it
// should go away.
type MethodInfo = {
    did: def_id,
    n_tps: uint,
    ident: ident,
    self_type: self_ty_
};

type Impl = { did: def_id, ident: ident, methods: ~[@MethodInfo] };
type ImplScope = @~[@Impl];
type ImplScopes = @list<ImplScope>;
type ImplMap = hashmap<node_id,ImplScopes>;

// Trait method resolution
type TraitMap = @hashmap<node_id,@dvec<def_id>>;

// Export mapping
type Export = { reexp: bool, id: def_id };
type ExportMap = hashmap<node_id, ~[Export]>;

enum PatternBindingMode {
    RefutableMode,
    IrrefutableMode
}

enum Namespace {
    ModuleNS,
    TypeNS,
    ValueNS,
    ImplNS
}

enum NamespaceResult {
    UnknownResult,
    UnboundResult,
    BoundResult(@Module, @NameBindings)
}

enum ImplNamespaceResult {
    UnknownImplResult,
    UnboundImplResult,
    BoundImplResult(@dvec<@Target>)
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

enum SelfBinding {
    NoSelfBinding,
    HasSelfBinding(node_id)
}

enum CaptureClause {
    NoCaptureClause,
    HasCaptureClause(capture_clause)
}

type ResolveVisitor = vt<()>;

enum ModuleDef {
    NoModuleDef,            // Does not define a module.
    ModuleDef(@Module),     // Defines a module.
}

/// Contains data for specific types of import directives.
enum ImportDirectiveSubclass {
    SingleImport(Atom /* target */, Atom /* source */),
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
    FunctionRibKind(node_id),

    // We passed through a class, impl, or trait and are now in one of its
    // methods. Allow references to ty params that that class, impl or trait
    // binds. Disallow any other upvars (including other ty params that are
    // upvars).
              // parent;   method itself
    MethodRibKind(node_id, MethodSort),

    // We passed through a function *item* scope. Disallow upvars.
    OpaqueFunctionRibKind
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

enum AllowCapturingSelfFlag {
    AllowCapturingSelf,         //< The "self" definition can be captured.
    DontAllowCapturingSelf,     //< The "self" definition cannot be captured.
}

enum EnumVariantOrConstResolution {
    FoundEnumVariant(def),
    FoundConst,
    EnumVariantOrConstNotFound
}

// FIXME (issue #2550): Should be a class but then it becomes not implicitly
// copyable due to a kind bug.

type Atom = uint;

fn Atom(n: uint) -> Atom {
    return n;
}

class AtomTable {
    let atoms: hashmap<@~str,Atom>;
    let strings: dvec<@~str>;
    let mut atom_count: uint;

    new() {
        self.atoms = hashmap::<@~str,Atom>(|x| str::hash(*x),
                                          |x, y| str::eq(*x, *y));
        self.strings = dvec();
        self.atom_count = 0u;
    }

    fn intern(string: @~str) -> Atom {
        match self.atoms.find(string) {
            none => { /* fall through */ }
            some(atom) => return atom
        }

        let atom = Atom(self.atom_count);
        self.atom_count += 1u;
        self.atoms.insert(string, atom);
        self.strings.push(string);

        return atom;
    }

    fn atom_to_str(atom: Atom) -> @~str {
        return self.strings.get_elt(atom);
    }

    fn atoms_to_strs(atoms: ~[Atom], f: fn(@~str) -> bool) {
        for atoms.each |atom| {
            if !f(self.atom_to_str(atom)) {
                return;
            }
        }
    }

    fn atoms_to_str(atoms: ~[Atom]) -> @~str {
        // XXX: str::connect should do this.
        let mut result = ~"";
        let mut first = true;
        for self.atoms_to_strs(atoms) |string| {
            if first {
                first = false;
            } else {
                result += ~"::";
            }

            result += *string;
        }

        // XXX: Shouldn't copy here. We need string builder functionality.
        return @result;
    }
}

/// Creates a hash table of atoms.
fn atom_hashmap<V:copy>() -> hashmap<Atom,V> {
    hashmap::<Atom,V>(uint::hash, uint::eq)
}

/**
 * One local scope. In Rust, local scopes can only contain value bindings.
 * Therefore, we don't have to worry about the other namespaces here.
 */
class Rib {
    let bindings: hashmap<Atom,def_like>;
    let kind: RibKind;

    new(kind: RibKind) {
        self.bindings = atom_hashmap();
        self.kind = kind;
    }
}

/// One import directive.
class ImportDirective {
    let module_path: @dvec<Atom>;
    let subclass: @ImportDirectiveSubclass;
    let span: span;

    new(module_path: @dvec<Atom>,
        subclass: @ImportDirectiveSubclass,
        span: span) {

        self.module_path = module_path;
        self.subclass = subclass;
        self.span = span;
    }
}

/// The item that an import resolves to.
class Target {
    let target_module: @Module;
    let bindings: @NameBindings;

    new(target_module: @Module, bindings: @NameBindings) {
        self.target_module = target_module;
        self.bindings = bindings;
    }
}

class ImportResolution {
    let span: span;

    // The number of outstanding references to this name. When this reaches
    // zero, outside modules can count on the targets being correct. Before
    // then, all bets are off; future imports could override this name.

    let mut outstanding_references: uint;

    let mut module_target: option<Target>;
    let mut value_target: option<Target>;
    let mut type_target: option<Target>;
    let mut impl_target: @dvec<@Target>;

    let mut used: bool;

    new(span: span) {
        self.span = span;

        self.outstanding_references = 0u;

        self.module_target = none;
        self.value_target = none;
        self.type_target = none;
        self.impl_target = @dvec();

        self.used = false;
    }

    fn target_for_namespace(namespace: Namespace) -> option<Target> {
        match namespace {
            ModuleNS    => return copy self.module_target,
            TypeNS      => return copy self.type_target,
            ValueNS     => return copy self.value_target,

            ImplNS => {
                if (*self.impl_target).len() > 0u {
                    return some(copy *(*self.impl_target).get_elt(0u));
                }
                return none;
            }
        }
    }
}

/// The link from a module up to its nearest parent node.
enum ParentLink {
    NoParentLink,
    ModuleParentLink(@Module, Atom),
    BlockParentLink(@Module, node_id)
}

/// One node in the tree of modules.
class Module {
    let parent_link: ParentLink;
    let mut def_id: option<def_id>;

    let children: hashmap<Atom,@NameBindings>;
    let imports: dvec<@ImportDirective>;

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

    let anonymous_children: hashmap<node_id,@Module>;

    // XXX: This is about to be reworked so that exports are on individual
    // items, not names.
    //
    // The atom is the name of the exported item, while the node ID is the
    // ID of the export path.

    let exported_names: hashmap<Atom,node_id>;

    // The status of resolving each import in this module.
    let import_resolutions: hashmap<Atom,@ImportResolution>;

    // The number of unresolved globs that this module exports.
    let mut glob_count: uint;

    // The index of the import we're resolving.
    let mut resolved_import_count: uint;

    // The list of implementation scopes, rooted from this module.
    let mut impl_scopes: ImplScopes;

    new(parent_link: ParentLink, def_id: option<def_id>) {
        self.parent_link = parent_link;
        self.def_id = def_id;

        self.children = atom_hashmap();
        self.imports = dvec();

        self.anonymous_children = int_hash();

        self.exported_names = atom_hashmap();

        self.import_resolutions = atom_hashmap();
        self.glob_count = 0u;
        self.resolved_import_count = 0u;

        self.impl_scopes = @nil;
    }

    fn all_imports_resolved() -> bool {
        return self.imports.len() == self.resolved_import_count;
    }
}

// XXX: This is a workaround due to is_none in the standard library mistakenly
// requiring a T:copy.

pure fn is_none<T>(x: option<T>) -> bool {
    match x {
        none => return true,
        some(_) => return false
    }
}

fn unused_import_lint_level(session: session) -> level {
    for session.opts.lint_opts.each |lint_option_pair| {
        let (lint_type, lint_level) = lint_option_pair;
        if lint_type == unused_imports {
            return lint_level;
        }
    }
    return allow;
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
class NameBindings {
    let mut module_def: ModuleDef;      //< Meaning in the module namespace.
    let mut type_def: option<def>;      //< Meaning in the type namespace.
    let mut value_def: option<def>;     //< Meaning in the value namespace.
    let mut impl_defs: ~[@Impl];        //< Meaning in the impl namespace.

    // For error reporting
    let mut module_span: option<span>;
    let mut type_span: option<span>;
    let mut value_span: option<span>;

    new() {
        self.module_def = NoModuleDef;
        self.type_def = none;
        self.value_def = none;
        self.impl_defs = ~[];
        self.module_span = none;
        self.type_span = none;
        self.value_span = none;
    }

    /// Creates a new module in this set of name bindings.
    fn define_module(parent_link: ParentLink, def_id: option<def_id>,
                     sp: span) {
        if self.module_def == NoModuleDef {
            let module_ = @Module(parent_link, def_id);
            self.module_def = ModuleDef(module_);
            self.module_span = some(sp);
        }
    }

    /// Records a type definition.
    fn define_type(def: def, sp: span) {
        self.type_def = some(def);
        self.type_span = some(sp);
    }

    /// Records a value definition.
    fn define_value(def: def, sp: span) {
        self.value_def = some(def);
        self.value_span = some(sp);
    }

    /// Records an impl definition.
    fn define_impl(implementation: @Impl) {
        self.impl_defs += ~[implementation];
    }

    /// Returns the module node if applicable.
    fn get_module_if_available() -> option<@Module> {
        match self.module_def {
            NoModuleDef         => return none,
            ModuleDef(module_)  => return some(module_)
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module() -> @Module {
        match self.module_def {
            NoModuleDef => {
                fail
                    ~"get_module called on a node with no module definition!";
            }
            ModuleDef(module_) => {
                return module_;
            }
        }
    }

    fn defined_in_namespace(namespace: Namespace) -> bool {
        match namespace {
            ModuleNS    => return self.module_def != NoModuleDef,
            TypeNS      => return self.type_def != none,
            ValueNS     => return self.value_def != none,
            ImplNS      => return self.impl_defs.len() >= 1u
        }
    }

    fn def_for_namespace(namespace: Namespace) -> option<def> {
        match namespace {
          TypeNS => return self.type_def,
          ValueNS => return self.value_def,
          ModuleNS => match self.module_def {
            NoModuleDef => return none,
            ModuleDef(module_) => match module_.def_id {
              none => return none,
              some(def_id) => return some(def_mod(def_id))
            }
          },
          ImplNS => {
            // Danger: Be careful what you use this for! def_ty is not
            // necessarily the right def.

            if self.impl_defs.len() == 0u {
                return none;
            }
            return some(def_ty(self.impl_defs[0].did));
          }
        }
    }

    fn span_for_namespace(namespace: Namespace) -> option<span> {
        match self.def_for_namespace(namespace) {
          some(d) => {
            match namespace {
              TypeNS   => self.type_span,
              ValueNS  => self.value_span,
              ModuleNS => self.module_span,
              _        => none
            }
          }
          none => none
        }
    }
}

/// Interns the names of the primitive types.
class PrimitiveTypeTable {
    let primitive_types: hashmap<Atom,prim_ty>;

    new(atom_table: @AtomTable) {
        self.primitive_types = atom_hashmap();

        self.intern(atom_table, @~"bool",    ty_bool);
        self.intern(atom_table, @~"char",    ty_int(ty_char));
        self.intern(atom_table, @~"float",   ty_float(ty_f));
        self.intern(atom_table, @~"f32",     ty_float(ty_f32));
        self.intern(atom_table, @~"f64",     ty_float(ty_f64));
        self.intern(atom_table, @~"int",     ty_int(ty_i));
        self.intern(atom_table, @~"i8",      ty_int(ty_i8));
        self.intern(atom_table, @~"i16",     ty_int(ty_i16));
        self.intern(atom_table, @~"i32",     ty_int(ty_i32));
        self.intern(atom_table, @~"i64",     ty_int(ty_i64));
        self.intern(atom_table, @~"str",     ty_str);
        self.intern(atom_table, @~"uint",    ty_uint(ty_u));
        self.intern(atom_table, @~"u8",      ty_uint(ty_u8));
        self.intern(atom_table, @~"u16",     ty_uint(ty_u16));
        self.intern(atom_table, @~"u32",     ty_uint(ty_u32));
        self.intern(atom_table, @~"u64",     ty_uint(ty_u64));
    }

    fn intern(atom_table: @AtomTable, string: @~str,
              primitive_type: prim_ty) {
        let atom = (*atom_table).intern(string);
        self.primitive_types.insert(atom, primitive_type);
    }
}

fn namespace_to_str(ns: Namespace) -> ~str {
    match ns {
      TypeNS   => ~"type",
      ValueNS  => ~"value",
      ModuleNS => ~"module",
      ImplNS   => ~"implementation"
    }
}

/// The main resolver class.
class Resolver {
    let session: session;
    let lang_items: LanguageItems;
    let crate: @crate;

    let atom_table: @AtomTable;

    let graph_root: @NameBindings;

    let unused_import_lint_level: level;

    let trait_info: hashmap<def_id,@hashmap<Atom,()>>;
    let structs: hashmap<def_id,bool>;

    // The number of imports that are currently unresolved.
    let mut unresolved_imports: uint;

    // The module that represents the current item scope.
    let mut current_module: @Module;

    // The current set of local scopes, for values.
    // XXX: Reuse ribs to avoid allocation.

    let value_ribs: @dvec<@Rib>;

    // The current set of local scopes, for types.
    let type_ribs: @dvec<@Rib>;

    // Whether the current context is an X-ray context. An X-ray context is
    // allowed to access private names of any module.
    let mut xray_context: XrayFlag;

    // The trait that the current context can refer to.
    let mut current_trait_refs: option<@dvec<def_id>>;

    // The atom for the keyword "self".
    let self_atom: Atom;

    // The atoms for the primitive types.
    let primitive_type_table: @PrimitiveTypeTable;

    // The four namespaces.
    let namespaces: ~[Namespace];

    let def_map: DefMap;
    let impl_map: ImplMap;
    let export_map: ExportMap;
    let trait_map: TraitMap;

    new(session: session, lang_items: LanguageItems, crate: @crate) {
        self.session = session;
        self.lang_items = copy lang_items;
        self.crate = crate;

        self.atom_table = @AtomTable();

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        self.graph_root = @NameBindings();
        (*self.graph_root).define_module(NoParentLink,
                                         some({ crate: 0, node: 0 }),
                                         crate.span);

        self.unused_import_lint_level = unused_import_lint_level(session);

        self.trait_info = new_def_hash();
        self.structs = new_def_hash();

        self.unresolved_imports = 0u;

        self.current_module = (*self.graph_root).get_module();
        self.value_ribs = @dvec();
        self.type_ribs = @dvec();

        self.xray_context = NoXray;
        self.current_trait_refs = none;

        self.self_atom = (*self.atom_table).intern(@~"self");
        self.primitive_type_table = @PrimitiveTypeTable(self.atom_table);

        self.namespaces = ~[ ModuleNS, TypeNS, ValueNS, ImplNS ];

        self.def_map = int_hash();
        self.impl_map = int_hash();
        self.export_map = int_hash();
        self.trait_map = @int_hash();
    }

    /// The main name resolution procedure.
    fn resolve(this: @Resolver) {
        self.build_reduced_graph(this);
        self.session.abort_if_errors();

        self.resolve_imports();
        self.session.abort_if_errors();

        self.record_exports();
        self.session.abort_if_errors();

        self.build_impl_scopes();
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
                                                      visitor)

            with *default_visitor()
        }));
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
    fn add_child(name: Atom,
                 reduced_graph_parent: ReducedGraphParent,
                 // Pass in the namespaces for the child item so that we can
                 // check for duplicate items in the same namespace
                 ns: ~[Namespace],
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
            none => {
              let child = @NameBindings();
              module_.children.insert(name, child);
              return (child, new_parent);
            }
            some(child) => {
              // We don't want to complain if the multiple definitions
              // are in different namespaces. (unless it's the impl namespace,
              // since impls can share a name)
              match ns.find(|n| n != ImplNS
                            && child.defined_in_namespace(n)) {
                some(ns) => {
                  self.session.span_err(sp,
                       #fmt("Duplicate definition of %s %s",
                            namespace_to_str(ns),
                            *(*self.atom_table).atom_to_str(name)));
                  do child.span_for_namespace(ns).iter() |sp| {
                      self.session.span_note(sp,
                           #fmt("First definition of %s %s here:",
                            namespace_to_str(ns),
                            *(*self.atom_table).atom_to_str(name)));
                  }
                }
                _ => {}
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

    fn get_parent_link(parent: ReducedGraphParent, name: Atom) -> ParentLink {
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

        let atom = (*self.atom_table).intern(item.ident);
        let sp = item.span;

        match item.node {
            item_mod(module_) => {
              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                       ~[ModuleNS], sp);

                let parent_link = self.get_parent_link(new_parent, atom);
                let def_id = { crate: 0, node: item.id };
              (*name_bindings).define_module(parent_link, some(def_id),
                                             sp);

                let new_parent =
                    ModuleReducedGraphParent((*name_bindings).get_module());

                visit_mod(module_, sp, item.id, new_parent, visitor);
            }
            item_foreign_mod(foreign_module) => {
              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                           ~[ModuleNS], sp);

                let parent_link = self.get_parent_link(new_parent, atom);
                let def_id = { crate: 0, node: item.id };
                (*name_bindings).define_module(parent_link, some(def_id),
                                               sp);

                let new_parent =
                    ModuleReducedGraphParent((*name_bindings).get_module());

                visit_item(item, new_parent, visitor);
            }

            // These items live in the value namespace.
            item_const(*) => {
              let (name_bindings, _) = self.add_child(atom, parent,
                                                      ~[ValueNS], sp);

                (*name_bindings).define_value(def_const(local_def(item.id)),
                                              sp);
            }
            item_fn(decl, _, _) => {
              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                        ~[ValueNS], sp);

                let def = def_fn(local_def(item.id), decl.purity);
                (*name_bindings).define_value(def, sp);
                visit_item(item, new_parent, visitor);
            }

            // These items live in the type namespace.
            item_ty(*) => {
              let (name_bindings, _) = self.add_child(atom, parent,
                                                      ~[TypeNS], sp);

                (*name_bindings).define_type(def_ty(local_def(item.id)), sp);
            }

            item_enum(enum_definition, _) => {

              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                               ~[TypeNS], sp);

                (*name_bindings).define_type(def_ty(local_def(item.id)), sp);

                for enum_definition.variants.each |variant| {
                    self.build_reduced_graph_for_variant(variant,
                                                         local_def(item.id),
                                                         new_parent,
                                                         visitor);
                }
            }

            // These items live in both the type and value namespaces.
            item_class(struct_definition, _) => {
                let (name_bindings, new_parent) =
                    match struct_definition.ctor {
                    none => {
                        let (name_bindings, new_parent) = self.add_child(atom,
                              parent, ~[TypeNS], sp);

                        (*name_bindings).define_type(def_ty(
                            local_def(item.id)), sp);
                        (name_bindings, new_parent)
                    }
                    some(ctor) => {
                        let (name_bindings, new_parent) = self.add_child(atom,
                                 parent, ~[ValueNS, TypeNS], sp);

                        (*name_bindings).define_type(def_ty(
                            local_def(item.id)), sp);

                        let purity = ctor.node.dec.purity;
                        let ctor_def = def_fn(local_def(ctor.node.id),
                                              purity);
                        (*name_bindings).define_value(ctor_def, sp);
                        (name_bindings, new_parent)
                    }
                };

                // Create the set of implementation information that the
                // implementation scopes (ImplScopes) need and write it into
                // the implementation definition list for this set of name
                // bindings.

                let mut method_infos = ~[];
                for struct_definition.members.each |class_member| {
                    match class_member.node {
                        class_method(method) => {
                            // XXX: Combine with impl method code below.
                            method_infos += ~[
                                @{
                                    did: local_def(method.id),
                                    n_tps: method.tps.len(),
                                    ident: method.ident,
                                    self_type: method.self_ty.node
                                }
                            ];
                        }
                        instance_var(*) => {
                            // Don't need to do anything with this.
                        }
                    }
                }

                let impl_info = @{
                    did: local_def(item.id),
                    ident: /* XXX: bad */ copy item.ident,
                    methods: method_infos
                };

                (*name_bindings).define_impl(impl_info);

                // Record the def ID of this struct.
                self.structs.insert(local_def(item.id),
                                    is_some(struct_definition.ctor));

                visit_item(item, new_parent, visitor);
            }

            item_impl(_, _, _, methods) => {
                // Create the set of implementation information that the
                // implementation scopes (ImplScopes) need and write it into
                // the implementation definition list for this set of name
                // bindings.
              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                               ~[ImplNS], sp);

                let mut method_infos = ~[];
                for methods.each |method| {
                    method_infos += ~[
                        @{
                            did: local_def(method.id),
                            n_tps: method.tps.len(),
                            ident: method.ident,
                            self_type: method.self_ty.node
                        }
                    ];
                }

                let impl_info = @{
                    did: local_def(item.id),
                    ident: /* XXX: bad */ copy item.ident,
                    methods: method_infos
                };

                (*name_bindings).define_impl(impl_info);
                visit_item(item, new_parent, visitor);
            }

          item_trait(_, _, methods) => {
              let (name_bindings, new_parent) = self.add_child(atom, parent,
                                                               ~[TypeNS], sp);

                // Add the names of all the methods to the trait info.
                let method_names = @atom_hashmap();
                for methods.each |method| {
                    let ty_m = trait_method_to_ty_method(method);

                    let atom = (*self.atom_table).intern(ty_m.ident);
                    // Add it to the trait info if not static,
                    // add it as a name in the enclosing module otherwise.
                    match ty_m.self_ty.node {
                      sty_static => {
                        // which parent to use??
                        let (method_name_bindings, _) =
                            self.add_child(atom, new_parent, ~[ValueNS],
                                           ty_m.span);
                        let def = def_static_method(local_def(ty_m.id),
                                                    ty_m.decl.purity);
                        (*method_name_bindings).define_value(def, ty_m.span);
                      }
                      _ => {
                        (*method_names).insert(atom, ());
                      }
                    }
                }

                let def_id = local_def(item.id);
                self.trait_info.insert(def_id, method_names);

                (*name_bindings).define_type(def_ty(def_id), sp);
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
                                       parent: ReducedGraphParent,
                                       &&visitor: vt<ReducedGraphParent>) {

        let atom = (*self.atom_table).intern(variant.node.name);
        let (child, _) = self.add_child(atom, parent, ~[ValueNS],
                                        variant.span);

        match variant.node.kind {
            tuple_variant_kind(_) => {
                (*child).define_value(def_variant(item_id,
                                                  local_def(variant.node.id)),
                                      variant.span);
            }
            struct_variant_kind(_) => {
                (*child).define_type(def_variant(item_id,
                                                 local_def(variant.node.id)),
                                     variant.span);
                self.structs.insert(local_def(variant.node.id), false);
            }
            enum_variant_kind(enum_definition) => {
                (*child).define_type(def_ty(local_def(variant.node.id)),
                                     variant.span);
                for enum_definition.variants.each |variant| {
                    self.build_reduced_graph_for_variant(variant, item_id,
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
        match view_item.node {
            view_item_import(view_paths) => {
                for view_paths.each |view_path| {
                    // Extract and intern the module part of the path. For
                    // globs and lists, the path is found directly in the AST;
                    // for simple paths we have to munge the path a little.

                    let module_path = @dvec();
                    match view_path.node {
                        view_path_simple(_, full_path, _) => {
                            let path_len = full_path.idents.len();
                            assert path_len != 0u;

                            for full_path.idents.eachi |i, ident| {
                                if i != path_len - 1u {
                                    let atom =
                                        (*self.atom_table).intern(ident);
                                    (*module_path).push(atom);
                                }
                            }
                        }

                        view_path_glob(module_ident_path, _) |
                        view_path_list(module_ident_path, _, _) => {
                            for module_ident_path.idents.each |ident| {
                                let atom = (*self.atom_table).intern(ident);
                                (*module_path).push(atom);
                            }
                        }
                    }

                    // Build up the import directives.
                    let module_ = self.get_module_from_parent(parent);
                    match view_path.node {
                        view_path_simple(binding, full_path, _) => {
                            let target_atom =
                                (*self.atom_table).intern(binding);
                            let source_ident = full_path.idents.last();
                            let source_atom =
                                (*self.atom_table).intern(source_ident);
                            let subclass = @SingleImport(target_atom,
                                                         source_atom);
                            self.build_import_directive(module_,
                                                        module_path,
                                                        subclass,
                                                        view_path.span);
                        }
                        view_path_list(_, source_idents, _) => {
                            for source_idents.each |source_ident| {
                                let name = source_ident.node.name;
                                let atom = (*self.atom_table).intern(name);
                                let subclass = @SingleImport(atom, atom);
                                self.build_import_directive(module_,
                                                            module_path,
                                                            subclass,
                                                            view_path.span);
                            }
                        }
                        view_path_glob(_, _) => {
                            self.build_import_directive(module_,
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
                        view_path_simple(ident, full_path, ident_id) => {
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

                            let atom = (*self.atom_table).intern(ident);
                            module_.exported_names.insert(atom, ident_id);
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
                                    let atom = (*self.atom_table).intern
                                        (path_list_ident.node.name);
                                    let id = path_list_ident.node.id;
                                    module_.exported_names.insert(atom, id);
                                }
                            }
                        }
                    }
                }
            }

            view_item_use(name, _, node_id) => {
                match find_use_stmt_cnum(self.session.cstore, node_id) {
                    some(crate_id) => {
                        let atom = (*self.atom_table).intern(name);
                        let (child_name_bindings, new_parent) =
                            // should this be in ModuleNS? --tjc
                            self.add_child(atom, parent, ~[ModuleNS],
                                           view_item.span);

                        let def_id = { crate: crate_id, node: 0 };
                        let parent_link = ModuleParentLink
                            (self.get_module_from_parent(new_parent), atom);

                        (*child_name_bindings).define_module(parent_link,
                                                             some(def_id),
                                                             view_item.span);
                        self.build_reduced_graph_for_external_crate
                            ((*child_name_bindings).get_module());
                    }
                    none => {
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

        let name = (*self.atom_table).intern(foreign_item.ident);

        match foreign_item.node {
            foreign_item_fn(fn_decl, type_parameters) => {
              let (name_bindings, new_parent) = self.add_child(name, parent,
                                              ~[ValueNS], foreign_item.span);

                let def = def_fn(local_def(foreign_item.id), fn_decl.purity);
                (*name_bindings).define_value(def, foreign_item.span);

                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters,
                                           foreign_item.id,
                                           0u,
                                           NormalRibKind)) || {

                    visit_foreign_item(foreign_item, new_parent, visitor);
                }
            }
        }

    }

    fn build_reduced_graph_for_block(block: blk,
                                     parent: ReducedGraphParent,
                                     &&visitor: vt<ReducedGraphParent>) {

        let mut new_parent;
        if self.block_needs_anonymous_module(block) {
            let block_id = block.node.id;

            debug!{"(building reduced graph for block) creating a new \
                    anonymous module for block %d",
                   block_id};

            let parent_module = self.get_module_from_parent(parent);
            let new_module = @Module(BlockParentLink(parent_module, block_id),
                                     none);
            parent_module.anonymous_children.insert(block_id, new_module);
            new_parent = ModuleReducedGraphParent(new_module);
        } else {
            new_parent = parent;
        }

        visit_block(block, new_parent, visitor);
    }

    fn handle_external_def(def: def, modules: hashmap<def_id, @Module>,
                           child_name_bindings: @NameBindings,
                           final_ident: ~str,
                           atom: Atom, new_parent: ReducedGraphParent) {
        match def {
          def_mod(def_id) | def_foreign_mod(def_id) => {
            match copy child_name_bindings.module_def {
              NoModuleDef => {
                debug!("(building reduced graph for \
                        external crate) building module \
                        %s", final_ident);
                let parent_link = self.get_parent_link(new_parent, atom);

                match modules.find(def_id) {
                  none => {
                    child_name_bindings.define_module(parent_link,
                                                      some(def_id),
                                                      dummy_sp());
                    modules.insert(def_id,
                                   child_name_bindings.get_module());
                  }
                  some(existing_module) => {
                    // Create an import resolution to
                    // avoid creating cycles in the
                    // module graph.

                    let resolution = @ImportResolution(dummy_sp());
                    resolution.outstanding_references = 0;

                    match existing_module.parent_link {
                      NoParentLink |
                      BlockParentLink(*) => {
                        fail ~"can't happen";
                      }
                      ModuleParentLink(parent_module, atom) => {

                        let name_bindings = parent_module.children.get(atom);

                        resolution.module_target =
                            some(Target(parent_module, name_bindings));
                      }
                    }

                    debug!("(building reduced graph for external crate) \
                            ... creating import resolution");

                    new_parent.import_resolutions.insert(atom, resolution);
                  }
                }
              }
              ModuleDef(module_) => {
                debug!("(building reduced graph for \
                        external crate) already created \
                        module");
                module_.def_id = some(def_id);
                modules.insert(def_id, module_);
              }
            }
          }
          def_fn(def_id, _) | def_static_method(def_id, _) |
          def_const(def_id) | def_variant(_, def_id) => {
            debug!("(building reduced graph for external \
                    crate) building value %s", final_ident);
            (*child_name_bindings).define_value(def, dummy_sp());
          }
          def_ty(def_id) => {
            debug!("(building reduced graph for external \
                    crate) building type %s", final_ident);

            // If this is a trait, add all the method names
            // to the trait info.

            match get_method_names_if_trait(self.session.cstore,
                                          def_id) {
              none => {
                // Nothing to do.
              }
              some(method_names) => {
                let interned_method_names = @atom_hashmap();
                for method_names.each |method_data| {
                    let (method_name, self_ty) = method_data;
                    debug!("(building reduced graph for \
                            external crate) ... adding \
                            trait method '%?'", method_name);

                    let m_atom = self.atom_table.intern(method_name);

                    // Add it to the trait info if not static.
                    if self_ty != sty_static {
                        interned_method_names.insert(m_atom, ());
                    }
                }
                self.trait_info.insert(def_id, interned_method_names);
              }
            }

            child_name_bindings.define_type(def, dummy_sp());
          }
          def_class(def_id, has_constructor) => {
            debug!("(building reduced graph for external \
                    crate) building type %s (value? %d)",
                   final_ident,
                   if has_constructor { 1 } else { 0 });
            child_name_bindings.define_type(def, dummy_sp());

            if has_constructor {
                child_name_bindings.define_value(def, dummy_sp());
            }
          }
          def_self(*) | def_arg(*) | def_local(*) |
          def_prim_ty(*) | def_ty_param(*) | def_binding(*) |
          def_use(*) | def_upvar(*) | def_region(*) |
          def_typaram_binder(*) => {
            fail fmt!("didn't expect `%?`", def);
          }
        }
    }

    /**
     * Builds the reduced graph rooted at the 'use' directive for an external
     * crate.
     */
    fn build_reduced_graph_for_external_crate(root: @Module) {
        let modules = new_def_hash();

        // Create all the items reachable by paths.
        for each_path(self.session.cstore, get(root.def_id).crate)
                |path_entry| {

            debug!{"(building reduced graph for external crate) found path \
                    entry: %s (%?)",
                   path_entry.path_string,
                   path_entry.def_like};

            let mut pieces = split_str(path_entry.path_string, ~"::");
            let final_ident = pop(pieces);

            // Find the module we need, creating modules along the way if we
            // need to.

            let mut current_module = root;
            for pieces.each |ident| {
                // Create or reuse a graph node for the child.
                let atom = (*self.atom_table).intern(@copy ident);
                let (child_name_bindings, new_parent) =
                    self.add_child(atom,
                                   ModuleReducedGraphParent(current_module),
                                   // May want a better span
                                   ~[], dummy_sp());

                // Define or reuse the module node.
                match child_name_bindings.module_def {
                    NoModuleDef => {
                        debug!{"(building reduced graph for external crate) \
                                autovivifying %s", ident};
                        let parent_link = self.get_parent_link(new_parent,
                                                               atom);
                        (*child_name_bindings).define_module(parent_link,
                                                       none, dummy_sp());
                    }
                    ModuleDef(_) => { /* Fall through. */ }
                }

                current_module = (*child_name_bindings).get_module();
            }

            // Add the new child item.
            let atom = (*self.atom_table).intern(@copy final_ident);
            let (child_name_bindings, new_parent) =
                self.add_child(atom,
                               ModuleReducedGraphParent(current_module),
                              ~[], dummy_sp());

            match path_entry.def_like {
                dl_def(def) => {
                    self.handle_external_def(def, modules,
                                             child_name_bindings,
                                             final_ident, atom, new_parent);
                }
                dl_impl(_) => {
                    // Because of the infelicitous way the metadata is
                    // written, we can't process this impl now. We'll get it
                    // later.

                    debug!{"(building reduced graph for external crate) \
                            ignoring impl %s", final_ident};
                }
                dl_field => {
                    debug!{"(building reduced graph for external crate) \
                            ignoring field %s", final_ident};
                }
            }
        }
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(module_: @Module,
                              module_path: @dvec<Atom>,
                              subclass: @ImportDirectiveSubclass,
                              span: span) {

        let directive = @ImportDirective(module_path, subclass, span);
        module_.imports.push(directive);

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        match *subclass {
            SingleImport(target, _) => {
                match module_.import_resolutions.find(target) {
                    some(resolution) => {
                        resolution.outstanding_references += 1u;
                    }
                    none => {
                        let resolution = @ImportResolution(span);
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
            debug!{"(resolving imports) iteration %u, %u imports left",
                   i, self.unresolved_imports};

            let module_root = (*self.graph_root).get_module();
            self.resolve_imports_for_module_subtree(module_root);

            if self.unresolved_imports == 0u {
                debug!{"(resolving imports) success"};
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
        debug!{"(resolving imports for module subtree) resolving %s",
               self.module_to_str(module_)};
        self.resolve_imports_for_module(module_);

        for module_.children.each |_name, child_node| {
            match (*child_node).get_module_if_available() {
                none => {
                    // Nothing to do.
                }
                some(child_module) => {
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
            debug!{"(resolving imports for module) all imports resolved for \
                   %s",
                   self.module_to_str(module_)};
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

        debug!{"(resolving import for module) resolving import `%s::...` in \
                `%s`",
               *(*self.atom_table).atoms_to_str((*module_path).get()),
               self.module_to_str(module_)};

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
                        SingleImport(target, source) => {
                            resolution_result =
                                self.resolve_single_import(module_,
                                                           containing_module,
                                                           target,
                                                           source);
                        }
                        GlobImport => {
                            let span = import_directive.span;
                            resolution_result =
                                self.resolve_glob_import(module_,
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

        if resolution_result != Indeterminate {
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

    fn resolve_single_import(module_: @Module, containing_module: @Module,
                             target: Atom, source: Atom)
                          -> ResolveResult<()> {

        debug!{"(resolving single import) resolving `%s` = `%s::%s` from \
                `%s`",
               *(*self.atom_table).atom_to_str(target),
               self.module_to_str(containing_module),
               *(*self.atom_table).atom_to_str(source),
               self.module_to_str(module_)};

        if !self.name_is_exported(containing_module, source) {
            debug!{"(resolving single import) name `%s` is unexported",
                   *(*self.atom_table).atom_to_str(source)};
            return Failed;
        }

        // We need to resolve all four namespaces for this to succeed.
        //
        // XXX: See if there's some way of handling namespaces in a more
        // generic way. We have four of them; it seems worth doing...

        let mut module_result = UnknownResult;
        let mut value_result = UnknownResult;
        let mut type_result = UnknownResult;
        let mut impl_result = UnknownImplResult;

        // Search for direct children of the containing module.
        match containing_module.children.find(source) {
            none => {
                // Continue.
            }
            some(child_name_bindings) => {
                if (*child_name_bindings).defined_in_namespace(ModuleNS) {
                    module_result = BoundResult(containing_module,
                                                child_name_bindings);
                }
                if (*child_name_bindings).defined_in_namespace(ValueNS) {
                    value_result = BoundResult(containing_module,
                                               child_name_bindings);
                }
                if (*child_name_bindings).defined_in_namespace(TypeNS) {
                    type_result = BoundResult(containing_module,
                                              child_name_bindings);
                }
                if (*child_name_bindings).defined_in_namespace(ImplNS) {
                    let targets = @dvec();
                    (*targets).push(@Target(containing_module,
                                            child_name_bindings));
                    impl_result = BoundImplResult(targets);
                }
            }
        }

        // Unless we managed to find a result in all four namespaces
        // (exceedingly unlikely), search imports as well.

        match (module_result, value_result, type_result, impl_result) {
            (BoundResult(*), BoundResult(*), BoundResult(*),
             BoundImplResult(*)) => {
                // Continue.
            }
            _ => {
                // If there is an unresolved glob at this point in the
                // containing module, bail out. We don't know enough to be
                // able to resolve this import.

                if containing_module.glob_count > 0u {
                    debug!{"(resolving single import) unresolved glob; \
                            bailing out"};
                    return Indeterminate;
                }

                // Now search the exported imports within the containing
                // module.

                match containing_module.import_resolutions.find(source) {
                    none => {
                        // The containing module definitely doesn't have an
                        // exported import with the name in question. We can
                        // therefore accurately report that the names are
                        // unbound.

                        if module_result == UnknownResult {
                            module_result = UnboundResult;
                        }
                        if value_result == UnknownResult {
                            value_result = UnboundResult;
                        }
                        if type_result == UnknownResult {
                            type_result = UnboundResult;
                        }
                        if impl_result == UnknownImplResult {
                            impl_result = UnboundImplResult;
                        }
                    }
                    some(import_resolution)
                            if import_resolution.outstanding_references
                                == 0u => {

                        fn get_binding(import_resolution: @ImportResolution,
                                       namespace: Namespace)
                                    -> NamespaceResult {

                            match (*import_resolution).
                                    target_for_namespace(namespace) {
                                none => {
                                    return UnboundResult;
                                }
                                some(target) => {
                                    import_resolution.used = true;
                                    return BoundResult(target.target_module,
                                                    target.bindings);
                                }
                            }
                        }

                        fn get_import_binding(import_resolution:
                                              @ImportResolution)
                                           -> ImplNamespaceResult {

                            if (*import_resolution.impl_target).len() == 0u {
                                return UnboundImplResult;
                            }
                            return BoundImplResult(import_resolution.
                                                impl_target);
                        }


                        // The name is an import which has been fully
                        // resolved. We can, therefore, just follow it.

                        if module_result == UnknownResult {
                            module_result = get_binding(import_resolution,
                                                        ModuleNS);
                        }
                        if value_result == UnknownResult {
                            value_result = get_binding(import_resolution,
                                                       ValueNS);
                        }
                        if type_result == UnknownResult {
                            type_result = get_binding(import_resolution,
                                                      TypeNS);
                        }
                        if impl_result == UnknownImplResult {
                            impl_result =
                                get_import_binding(import_resolution);
                        }
                    }
                    some(_) => {
                        // The import is unresolved. Bail out.
                        debug!{"(resolving single import) unresolved import; \
                                bailing out"};
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
                debug!{"(resolving single import) found module binding"};
                import_resolution.module_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult => {
                debug!{"(resolving single import) didn't find module \
                        binding"};
            }
            UnknownResult => {
                fail ~"module result should be known at this point";
            }
        }
        match value_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.value_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail ~"value result should be known at this point";
            }
        }
        match type_result {
            BoundResult(target_module, name_bindings) => {
                import_resolution.type_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult => { /* Continue. */ }
            UnknownResult => {
                fail ~"type result should be known at this point";
            }
        }
        match impl_result {
            BoundImplResult(targets) => {
                for (*targets).each |target| {
                    (*import_resolution.impl_target).push(target);
                }
            }
            UnboundImplResult => { /* Continue. */ }
            UnknownImplResult => {
                fail ~"impl result should be known at this point";
            }
        }

        let i = import_resolution;
        match (i.module_target, i.value_target,
               i.type_target, i.impl_target) {
          /*
            If this name wasn't found in any of the four namespaces, it's
            definitely unresolved
           */
          (none, none, none, v) if v.len() == 0 => { return Failed; }
          _ => {}
        }

        assert import_resolution.outstanding_references >= 1u;
        import_resolution.outstanding_references -= 1u;

        debug!{"(resolving single import) successfully resolved import"};
        return Success(());
    }

    /**
     * Resolves a glob import. Note that this function cannot fail; it either
     * succeeds or bails out (as importing * from an empty module or a module
     * that exports nothing is valid).
     */
    fn resolve_glob_import(module_: @Module,
                           containing_module: @Module,
                           span: span)
                        -> ResolveResult<()> {

        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).

        if !(*containing_module).all_imports_resolved() {
            debug!{"(resolving glob import) target module has unresolved \
                    imports; bailing out"};
            return Indeterminate;
        }

        assert containing_module.glob_count == 0u;

        // Add all resolved imports from the containing module.
        for containing_module.import_resolutions.each
                |atom, target_import_resolution| {

            if !self.name_is_exported(containing_module, atom) {
                debug!{"(resolving glob import) name `%s` is unexported",
                       *(*self.atom_table).atom_to_str(atom)};
                again;
            }

            debug!{"(resolving glob import) writing module resolution \
                    %? into `%s`",
                   is_none(target_import_resolution.module_target),
                   self.module_to_str(module_)};

            // Here we merge two import resolutions.
            match module_.import_resolutions.find(atom) {
                none => {
                    // Simple: just copy the old import resolution.
                    let new_import_resolution =
                        @ImportResolution(target_import_resolution.span);
                    new_import_resolution.module_target =
                        copy target_import_resolution.module_target;
                    new_import_resolution.value_target =
                        copy target_import_resolution.value_target;
                    new_import_resolution.type_target =
                        copy target_import_resolution.type_target;
                    new_import_resolution.impl_target =
                        copy target_import_resolution.impl_target;

                    module_.import_resolutions.insert
                        (atom, new_import_resolution);
                }
                some(dest_import_resolution) => {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    match copy target_import_resolution.module_target {
                        none => {
                            // Continue.
                        }
                        some(module_target) => {
                            dest_import_resolution.module_target =
                                some(copy module_target);
                        }
                    }
                    match copy target_import_resolution.value_target {
                        none => {
                            // Continue.
                        }
                        some(value_target) => {
                            dest_import_resolution.value_target =
                                some(copy value_target);
                        }
                    }
                    match copy target_import_resolution.type_target {
                        none => {
                            // Continue.
                        }
                        some(type_target) => {
                            dest_import_resolution.type_target =
                                some(copy type_target);
                        }
                    }
                    if (*target_import_resolution.impl_target).len() > 0u &&
                            !ptr_eq(target_import_resolution.impl_target,
                                    dest_import_resolution.impl_target) {
                        for (*target_import_resolution.impl_target).each
                                |impl_target| {

                            (*dest_import_resolution.impl_target).
                                push(impl_target);

                        }
                    }
                }
            }
        }

        // Add all children from the containing module.
        for containing_module.children.each |atom, name_bindings| {
            if !self.name_is_exported(containing_module, atom) {
                debug!{"(resolving glob import) name `%s` is unexported",
                       *(*self.atom_table).atom_to_str(atom)};
                again;
            }

            let mut dest_import_resolution;
            match module_.import_resolutions.find(atom) {
                none => {
                    // Create a new import resolution from this child.
                    dest_import_resolution = @ImportResolution(span);
                    module_.import_resolutions.insert
                        (atom, dest_import_resolution);
                }
                some(existing_import_resolution) => {
                    dest_import_resolution = existing_import_resolution;
                }
            }


            debug!{"(resolving glob import) writing resolution `%s` in `%s` \
                    to `%s`",
                   *(*self.atom_table).atom_to_str(atom),
                   self.module_to_str(containing_module),
                   self.module_to_str(module_)};

            // Merge the child item into the import resolution.
            if (*name_bindings).defined_in_namespace(ModuleNS) {
                debug!{"(resolving glob import) ... for module target"};
                dest_import_resolution.module_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(ValueNS) {
                debug!{"(resolving glob import) ... for value target"};
                dest_import_resolution.value_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(TypeNS) {
                debug!{"(resolving glob import) ... for type target"};
                dest_import_resolution.type_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(ImplNS) {
                debug!{"(resolving glob import) ... for impl target"};
                (*dest_import_resolution.impl_target).push
                    (@Target(containing_module, name_bindings));
            }
        }

        debug!{"(resolving glob import) successfully resolved import"};
        return Success(());
    }

    fn resolve_module_path_from_root(module_: @Module,
                                     module_path: @dvec<Atom>,
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
            match self.resolve_name_in_module(search_module, name, ModuleNS,
                                            xray) {

                Failed => {
                    self.session.span_err(span, ~"unresolved name");
                    return Failed;
                }
                Indeterminate => {
                    debug!{"(resolving module path for import) module \
                            resolution is indeterminate: %s",
                            *(*self.atom_table).atom_to_str(name)};
                    return Indeterminate;
                }
                Success(target) => {
                    match target.bindings.module_def {
                        NoModuleDef => {
                            // Not a module.
                            self.session.span_err(span,
                                                  fmt!{"not a module: %s",
                                                       *(*self.atom_table).
                                                         atom_to_str(name)});
                            return Failed;
                        }
                        ModuleDef(module_) => {
                            search_module = module_;
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
                                      module_path: @dvec<Atom>,
                                      xray: XrayFlag,
                                      span: span)
                                   -> ResolveResult<@Module> {

        let module_path_len = (*module_path).len();
        assert module_path_len > 0u;

        debug!{"(resolving module path for import) processing `%s` rooted at \
               `%s`",
               *(*self.atom_table).atoms_to_str((*module_path).get()),
               self.module_to_str(module_)};

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
                debug!{"(resolving module path for import) indeterminate; \
                        bailing"};
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
                                     name: Atom,
                                     namespace: Namespace)
                                  -> ResolveResult<Target> {

        debug!{"(resolving item in lexical scope) resolving `%s` in \
                namespace %? in `%s`",
               *(*self.atom_table).atom_to_str(name),
               namespace,
               self.module_to_str(module_)};

        // The current module node is handled specially. First, check for
        // its immediate children.

        match module_.children.find(name) {
            some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) => {

                return Success(Target(module_, name_bindings));
            }
            some(_) | none => { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.

        match module_.import_resolutions.find(name) {
            none => {
                // Not found; continue.
            }
            some(import_resolution) => {
                match (*import_resolution).target_for_namespace(namespace) {
                    none => {
                        // Not found; continue.
                        debug!{"(resolving item in lexical scope) found \
                                import resolution, but not in namespace %?",
                               namespace};
                    }
                    some(target) => {
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
                    debug!{"(resolving item in lexical scope) unresolved \
                            module"};
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

                    debug!{"(resolving item in lexical scope) indeterminate \
                            higher scope; bailing"};
                    return Indeterminate;
                }
                Success(target) => {
                    // We found the module.
                    return Success(copy target);
                }
            }
        }
    }

    fn resolve_module_in_lexical_scope(module_: @Module, name: Atom)
                                    -> ResolveResult<@Module> {

        match self.resolve_item_in_lexical_scope(module_, name, ModuleNS) {
            Success(target) => {
                match target.bindings.module_def {
                    NoModuleDef => {
                        error!{"!!! (resolving module in lexical scope) module
                                wasn't actually a module!"};
                        return Failed;
                    }
                    ModuleDef(module_) => {
                        return Success(module_);
                    }
                }
            }
            Indeterminate => {
                debug!{"(resolving module in lexical scope) indeterminate; \
                        bailing"};
                return Indeterminate;
            }
            Failed => {
                debug!{"(resolving module in lexical scope) failed to \
                        resolve"};
                return Failed;
            }
        }
    }

    fn name_is_exported(module_: @Module, name: Atom) -> bool {
        return module_.exported_names.size() == 0u ||
                module_.exported_names.contains_key(name);
    }

    /**
     * Attempts to resolve the supplied name in the given module for the
     * given namespace. If successful, returns the target corresponding to
     * the name.
     */
    fn resolve_name_in_module(module_: @Module,
                              name: Atom,
                              namespace: Namespace,
                              xray: XrayFlag)
                           -> ResolveResult<Target> {

        debug!{"(resolving name in module) resolving `%s` in `%s`",
               *(*self.atom_table).atom_to_str(name),
               self.module_to_str(module_)};

        if xray == NoXray && !self.name_is_exported(module_, name) {
            debug!{"(resolving name in module) name `%s` is unexported",
                   *(*self.atom_table).atom_to_str(name)};
            return Failed;
        }

        // First, check the direct children of the module.
        match module_.children.find(name) {
            some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) => {

                debug!{"(resolving name in module) found node as child"};
                return Success(Target(module_, name_bindings));
            }
            some(_) | none => {
                // Continue.
            }
        }

        // Next, check the module's imports. If the module has a glob, then
        // we bail out; we don't know its imports yet.

        if module_.glob_count > 0u {
            debug!{"(resolving name in module) module has glob; bailing out"};
            return Indeterminate;
        }

        // Otherwise, we check the list of resolved imports.
        match module_.import_resolutions.find(name) {
            some(import_resolution) => {
                if import_resolution.outstanding_references != 0u {
                    debug!{"(resolving name in module) import unresolved; \
                            bailing out"};
                    return Indeterminate;
                }

                match (*import_resolution).target_for_namespace(namespace) {
                    none => {
                        debug!{"(resolving name in module) name found, but \
                                not in namespace %?",
                               namespace};
                    }
                    some(target) => {
                        debug!{"(resolving name in module) resolved to \
                                import"};
                        import_resolution.used = true;
                        return Success(copy target);
                    }
                }
            }
            none => {
                // Continue.
            }
        }

        // We're out of luck.
        debug!{"(resolving name in module) failed to resolve %s",
               *(*self.atom_table).atom_to_str(name)};
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
        match *import_directive.subclass {
            SingleImport(target, source) => {
                target_name = target;
                source_name = source;
            }
            GlobImport => {
                fail ~"found `import *`, which is invalid";
            }
        }

        debug!{"(resolving one-level naming result) resolving import `%s` = \
                `%s` in `%s`",
                *(*self.atom_table).atom_to_str(target_name),
                *(*self.atom_table).atom_to_str(source_name),
                self.module_to_str(module_)};

        // Find the matching items in the lexical scope chain for every
        // namespace. If any of them come back indeterminate, this entire
        // import is indeterminate.

        let mut module_result;
        debug!{"(resolving one-level naming result) searching for module"};
        match self.resolve_item_in_lexical_scope(module_,
                                               source_name,
                                               ModuleNS) {

            Failed => {
                debug!{"(resolving one-level renaming import) didn't find \
                        module result"};
                module_result = none;
            }
            Indeterminate => {
                debug!{"(resolving one-level renaming import) module result \
                        is indeterminate; bailing"};
                return Indeterminate;
            }
            Success(name_bindings) => {
                debug!{"(resolving one-level renaming import) module result \
                        found"};
                module_result = some(copy name_bindings);
            }
        }

        let mut value_result;
        debug!{"(resolving one-level naming result) searching for value"};
        match self.resolve_item_in_lexical_scope(module_,
                                               source_name,
                                               ValueNS) {

            Failed => {
                debug!{"(resolving one-level renaming import) didn't find \
                        value result"};
                value_result = none;
            }
            Indeterminate => {
                debug!{"(resolving one-level renaming import) value result \
                        is indeterminate; bailing"};
                return Indeterminate;
            }
            Success(name_bindings) => {
                debug!{"(resolving one-level renaming import) value result \
                        found"};
                value_result = some(copy name_bindings);
            }
        }

        let mut type_result;
        debug!{"(resolving one-level naming result) searching for type"};
        match self.resolve_item_in_lexical_scope(module_,
                                               source_name,
                                               TypeNS) {

            Failed => {
                debug!{"(resolving one-level renaming import) didn't find \
                        type result"};
                type_result = none;
            }
            Indeterminate => {
                debug!{"(resolving one-level renaming import) type result is \
                        indeterminate; bailing"};
                return Indeterminate;
            }
            Success(name_bindings) => {
                debug!{"(resolving one-level renaming import) type result \
                        found"};
                type_result = some(copy name_bindings);
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

        let mut impl_result;
        debug!{"(resolving one-level naming result) searching for impl"};
        match self.resolve_item_in_lexical_scope(module_,
                                               source_name,
                                               ImplNS) {

            Failed => {
                debug!{"(resolving one-level renaming import) didn't find \
                        impl result"};
                impl_result = none;
            }
            Indeterminate => {
                debug!{"(resolving one-level renaming import) impl result is \
                        indeterminate; bailing"};
                return Indeterminate;
            }
            Success(name_bindings) => {
                debug!{"(resolving one-level renaming import) impl result \
                        found"};
                impl_result = some(@copy name_bindings);
            }
        }

        // If nothing at all was found, that's an error.
        if is_none(module_result) && is_none(value_result) &&
                is_none(type_result) && is_none(impl_result) {

            self.session.span_err(import_directive.span,
                                  ~"unresolved import");
            return Failed;
        }

        // Otherwise, proceed and write in the bindings.
        match module_.import_resolutions.find(target_name) {
            none => {
                fail ~"(resolving one-level renaming import) reduced graph \
                      construction or glob importing should have created the \
                      import resolution name by now";
            }
            some(import_resolution) => {
                debug!{"(resolving one-level renaming import) writing module \
                        result %? for `%s` into `%s`",
                       is_none(module_result),
                       *(*self.atom_table).atom_to_str(target_name),
                       self.module_to_str(module_)};

                import_resolution.module_target = module_result;
                import_resolution.value_target = value_result;
                import_resolution.type_target = type_result;

                match impl_result {
                    none => {
                        // Nothing to do.
                    }
                    some(impl_result) => {
                        (*import_resolution.impl_target).push(impl_result);
                    }
                }

                assert import_resolution.outstanding_references >= 1u;
                import_resolution.outstanding_references -= 1u;
            }
        }

        debug!{"(resolving one-level renaming import) successfully resolved"};
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
            match (*child_node).get_module_if_available() {
                none => {
                    // Continue.
                }
                some(child_module) => {
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
            some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
            }
            none => {
                // Record exports for the root module.
            }
            some(_) => {
                // Bail out.
                debug!{"(recording exports for module subtree) not recording \
                        exports for `%s`",
                       self.module_to_str(module_)};
                return;
            }
        }

        self.record_exports_for_module(module_);

        for module_.children.each |_atom, child_name_bindings| {
            match (*child_name_bindings).get_module_if_available() {
                none => {
                    // Nothing to do.
                }
                some(child_module) => {
                    self.record_exports_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each |_node_id, child_module| {
            self.record_exports_for_module_subtree(child_module);
        }
    }

    fn record_exports_for_module(module_: @Module) {
        for module_.exported_names.each |name, node_id| {
            let mut exports = ~[];
            for self.namespaces.each |namespace| {
                // Ignore impl namespaces; they cause the original resolve
                // to fail.

                if namespace == ImplNS {
                    again;
                }

                match self.resolve_definition_of_name_in_module(module_,
                                                              name,
                                                              namespace,
                                                              Xray) {
                    NoNameDefinition => {
                        // Nothing to do.
                    }
                    ChildNameDefinition(target_def) => {
                        vec::push(exports, {
                            reexp: false,
                            id: def_id_of_def(target_def)
                        });
                    }
                    ImportNameDefinition(target_def) => {
                        vec::push(exports, {
                            reexp: true,
                            id: def_id_of_def(target_def)
                        });
                    }
                }
            }

            self.export_map.insert(node_id, exports);
        }
    }

    // Implementation scope creation
    //
    // This is a fairly simple pass that simply gathers up all the typeclass
    // implementations in scope and threads a series of singly-linked series
    // of impls through the tree.

    fn build_impl_scopes() {
        let root_module = (*self.graph_root).get_module();
        self.build_impl_scopes_for_module_subtree(root_module);
    }

    fn build_impl_scopes_for_module_subtree(module_: @Module) {
        // If this isn't a local crate, then bail out. We don't need to
        // resolve implementations for external crates.

        match module_.def_id {
            some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
            }
            none => {
                // Resolve implementation scopes for the root module.
            }
            some(_) => {
                // Bail out.
                debug!{"(building impl scopes for module subtree) not \
                        resolving implementations for `%s`",
                       self.module_to_str(module_)};
                return;
            }
        }

        self.build_impl_scope_for_module(module_);

        for module_.children.each |_atom, child_name_bindings| {
            match (*child_name_bindings).get_module_if_available() {
                none => {
                    // Nothing to do.
                }
                some(child_module) => {
                    self.build_impl_scopes_for_module_subtree(child_module);
                }
            }
        }

        for module_.anonymous_children.each |_node_id, child_module| {
            self.build_impl_scopes_for_module_subtree(child_module);
        }
    }

    fn build_impl_scope_for_module(module_: @Module) {
        let mut impl_scope = ~[];

        debug!{"(building impl scope for module) processing module %s (%?)",
               self.module_to_str(module_),
               copy module_.def_id};

        // Gather up all direct children implementations in the module.
        for module_.children.each |_impl_name, child_name_bindings| {
            if child_name_bindings.impl_defs.len() >= 1u {
                impl_scope += child_name_bindings.impl_defs;
            }
        }

        debug!{"(building impl scope for module) found %u impl(s) as direct \
                children",
               impl_scope.len()};

        // Gather up all imports.
        for module_.import_resolutions.each |_impl_name, import_resolution| {
            for (*import_resolution.impl_target).each |impl_target| {
                debug!{"(building impl scope for module) found impl def"};
                impl_scope += impl_target.bindings.impl_defs;
            }
        }

        debug!{"(building impl scope for module) found %u impl(s) in total",
               impl_scope.len()};

        // Determine the parent's implementation scope.
        let mut parent_impl_scopes;
        match module_.parent_link {
            NoParentLink => {
                parent_impl_scopes = @nil;
            }
            ModuleParentLink(parent_module_node, _) |
            BlockParentLink(parent_module_node, _) => {
                parent_impl_scopes = parent_module_node.impl_scopes;
            }
        }

        // Create the new implementation scope, if it was nonempty, and chain
        // it up to the parent.

        if impl_scope.len() >= 1u {
            module_.impl_scopes = @cons(@impl_scope, parent_impl_scopes);
        } else {
            module_.impl_scopes = parent_impl_scopes;
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

    fn with_scope(name: option<Atom>, f: fn()) {
        let orig_module = self.current_module;

        // Move down in the graph.
        match name {
            none => {
                // Nothing to do.
            }
            some(name) => {
                match orig_module.children.find(name) {
                    none => {
                        debug!{"!!! (with scope) didn't find `%s` in `%s`",
                               *(*self.atom_table).atom_to_str(name),
                               self.module_to_str(orig_module)};
                    }
                    some(name_bindings) => {
                        match (*name_bindings).get_module_if_available() {
                            none => {
                                debug!{"!!! (with scope) didn't find module \
                                        for `%s` in `%s`",
                                       *(*self.atom_table).atom_to_str(name),
                                       self.module_to_str(orig_module)};
                            }
                            some(module_) => {
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

    fn upvarify(ribs: @dvec<@Rib>, rib_index: uint, def_like: def_like,
                span: span, allow_capturing_self: AllowCapturingSelfFlag)
             -> option<def_like> {

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
                return some(def_like);
            }
        }

        let mut rib_index = rib_index + 1u;
        while rib_index < (*ribs).len() {
            let rib = (*ribs).get_elt(rib_index);
            match rib.kind {
                NormalRibKind => {
                    // Nothing to do. Continue.
                }
                FunctionRibKind(function_id) => {
                    if !is_ty_param {
                        def = def_upvar(def_id_of_def(def).node,
                                        @def,
                                        function_id);
                    }
                }
                MethodRibKind(item_id, method_id) => {
                  // If the def is a ty param, and came from the parent
                  // item, it's ok
                  match def {
                    def_ty_param(did, _) if self.def_map.find(copy(did.node))
                      == some(def_typaram_binder(item_id)) => {
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

                    return none;
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

                    return none;
                }
            }

            rib_index += 1u;
        }

        return some(dl_def(def));
    }

    fn search_ribs(ribs: @dvec<@Rib>, name: Atom, span: span,
                   allow_capturing_self: AllowCapturingSelfFlag)
                -> option<def_like> {

        // XXX: This should not use a while loop.
        // XXX: Try caching?

        let mut i = (*ribs).len();
        while i != 0u {
            i -= 1u;
            let rib = (*ribs).get_elt(i);
            match rib.bindings.find(name) {
                some(def_like) => {
                    return self.upvarify(ribs, i, def_like, span,
                                      allow_capturing_self);
                }
                none => {
                    // Continue.
                }
            }
        }

        return none;
    }

    // XXX: This shouldn't be unsafe!
    fn resolve_crate() unsafe {
        debug!{"(resolving crate) starting"};

        // To avoid a failure in metadata encoding later, we have to add the
        // crate-level implementation scopes

        self.impl_map.insert(0, (*self.graph_root).get_module().impl_scopes);

        // XXX: This is awful!
        let this = ptr::addr_of(self);
        visit_crate(*self.crate, (), mk_vt(@{
            visit_item: |item, _context, visitor|
                (*this).resolve_item(item, visitor),
            visit_arm: |arm, _context, visitor|
                (*this).resolve_arm(arm, visitor),
            visit_block: |block, _context, visitor|
                (*this).resolve_block(block, visitor),
            visit_expr: |expr, _context, visitor|
                (*this).resolve_expr(expr, visitor),
            visit_local: |local, _context, visitor|
                (*this).resolve_local(local, visitor),
            visit_ty: |ty, _context, visitor|
                (*this).resolve_type(ty, visitor)
            with *default_visitor()
        }));
    }

    fn resolve_item(item: @item, visitor: ResolveVisitor) {
        debug!{"(resolving item) resolving %s", *item.ident};

        // Items with the !resolve_unexported attribute are X-ray contexts.
        // This is used to allow the test runner to run unexported tests.
        let orig_xray_flag = self.xray_context;
        if contains_name(attr_metas(item.attrs), ~"!resolve_unexported") {
            self.xray_context = Xray;
        }

        match item.node {
            item_enum(_, type_parameters) |
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
                self_type_rib.bindings.insert(self.self_atom,
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
                            none =>
                                self.session.span_err(trt.path.span,
                                                      ~"attempt to derive a \
                                                       nonexistent trait"),
                            some(def) => {
                                // Write a mapping from the trait ID to the
                                // definition of the trait into the definition
                                // map.

                                debug!{"(resolving trait) found trait def: \
                                       %?", def};

                                self.record_def(trt.ref_id, def);
                            }
                        }
                    }

                    for methods.each |method| {
                        // Create a new rib for the method-specific type
                        // parameters.
                        //
                        // XXX: Do we need a node ID here?

                        match method {
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
                                   struct_def.members,
                                   struct_def.ctor,
                                   struct_def.dtor,
                                   visitor);
            }

            item_mod(module_) => {
                let atom = (*self.atom_table).intern(item.ident);
                do self.with_scope(some(atom)) {
                    self.resolve_module(module_, item.span, item.ident,
                                        item.id, visitor);
                }
            }

            item_foreign_mod(foreign_module) => {
                let atom = (*self.atom_table).intern(item.ident);
                do self.with_scope(some(atom)) {
                    for foreign_module.items.each |foreign_item| {
                        match foreign_item.node {
                            foreign_item_fn(_, type_parameters) => {
                                do self.with_type_parameter_rib
                                    (HasTypeParameters(&type_parameters,
                                                       foreign_item.id,
                                                       0u,
                                                       OpaqueFunctionRibKind))
                                        || {

                                    visit_foreign_item(foreign_item, (),
                                                       visitor);
                                }
                            }
                        }
                    }
                }
            }

            item_fn(fn_decl, ty_params, block) => {
                // If this is the main function, we must record it in the
                // session.
                //
                // For speed, we put the string comparison last in this chain
                // of conditionals.

                if !self.session.building_library &&
                        is_none(self.session.main_fn) &&
                        *item.ident == ~"main" {

                    self.session.main_fn = some((item.id, item.span));
                }

                self.resolve_function(OpaqueFunctionRibKind,
                                      some(@fn_decl),
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
                visit_item(item, (), visitor);
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
                    let name =
                        (*self.atom_table).intern(type_parameter.ident);
                    debug!{"with_type_parameter_rib: %d %d", node_id,
                           type_parameter.id};
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
            HasTypeParameters(type_parameters, _, _, _) => {
                (*self.type_ribs).pop();
            }

            NoTypeParameters =>{
                // Nothing to do.
            }
        }
    }

    fn resolve_function(rib_kind: RibKind,
                        optional_declaration: option<@fn_decl>,
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
                        none => {
                            self.session.span_err(capture_item.span,
                                                  ~"unresolved name in \
                                                   capture clause");
                        }
                        some(def) => {
                            self.record_def(capture_item.id, def);
                        }
                    }
                }
            }
        }

        // Create a value rib for the function.
        let function_value_rib = @Rib(rib_kind);
        (*self.value_ribs).push(function_value_rib);

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
                    (*function_value_rib).bindings.insert(self.self_atom,
                                                          def_like);
                }
            }

            // Add each argument to the rib.
            match optional_declaration {
                none => {
                    // Nothing to do.
                }
                some(declaration) => {
                    for declaration.inputs.each |argument| {
                        let name = (*self.atom_table).intern(argument.ident);
                        let def_like = dl_def(def_arg(argument.id,
                                                      argument.mode));
                        (*function_value_rib).bindings.insert(name, def_like);

                        self.resolve_type(argument.ty, visitor);

                        debug!{"(resolving function) recorded argument `%s`",
                               *(*self.atom_table).atom_to_str(name)};
                    }

                    self.resolve_type(declaration.output, visitor);
                }
            }

            // Resolve the function body.
            self.resolve_block(block, visitor);

            debug!{"(resolving function) leaving function"};
        }

        (*self.value_ribs).pop();
    }

    fn resolve_type_parameters(type_parameters: ~[ty_param],
                               visitor: ResolveVisitor) {

        for type_parameters.each |type_parameter| {
            for (*type_parameter.bounds).each |bound| {
                match bound {
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
                     class_members: ~[@class_member],
                     optional_constructor: option<class_ctor>,
                     optional_destructor: option<class_dtor>,
                     visitor: ResolveVisitor) {

        // Add a type into the def map. This is needed to prevent an ICE in
        // ty::impl_traits.

        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = (*type_parameters).len();
        let borrowed_type_parameters: &~[ty_param] = &*type_parameters;
        do self.with_type_parameter_rib(HasTypeParameters
                                        (borrowed_type_parameters, id, 0u,
                                         NormalRibKind)) {

            // Resolve the type parameters.
            self.resolve_type_parameters(*type_parameters, visitor);

            // Resolve implemented traits.
            for traits.each |trt| {
                match self.resolve_path(trt.path, TypeNS, true, visitor) {
                    none => {
                        self.session.span_err(trt.path.span,
                                              ~"attempt to implement a \
                                               nonexistent trait");
                    }
                    some(def) => {
                        // Write a mapping from the trait ID to the
                        // definition of the trait into the definition
                        // map.

                        debug!{"(resolving class) found trait def: %?", def};

                        self.record_def(trt.ref_id, def);

                        // XXX: This is wrong but is needed for tests to
                        // pass.

                        self.record_def(id, def);
                    }
                }
            }

            // Resolve methods.
            for class_members.each |class_member| {
                match class_member.node {
                    class_method(method) => {
                      self.resolve_method(MethodRibKind(id,
                                               Provided(method.id)),
                                          method,
                                          outer_type_parameter_count,
                                          visitor);
                    }
                    instance_var(_, field_type, _, _, _) => {
                        self.resolve_type(field_type, visitor);
                    }
                }
            }

            // Resolve the constructor, if applicable.
            match optional_constructor {
                none => {
                    // Nothing to do.
                }
                some(constructor) => {
                    self.resolve_function(NormalRibKind,
                                          some(@constructor.node.dec),
                                          NoTypeParameters,
                                          constructor.node.body,
                                          HasSelfBinding(constructor.node.
                                                         self_id),
                                          NoCaptureClause,
                                          visitor);
                }
            }

            // Resolve the destructor, if applicable.
            match optional_destructor {
                none => {
                    // Nothing to do.
                }
                some(destructor) => {
                    self.resolve_function(NormalRibKind,
                                          none,
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
                              some(@method.decl),
                              type_parameters,
                              method.body,
                              self_binding,
                              NoCaptureClause,
                              visitor);
    }

    fn resolve_implementation(id: node_id,
                              span: span,
                              type_parameters: ~[ty_param],
                              trait_references: ~[@trait_ref],
                              self_type: @ty,
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
            if trait_references.len() >= 1 {
                let mut new_trait_refs = @dvec();
                for trait_references.each |trait_reference| {
                    match self.resolve_path(
                        trait_reference.path, TypeNS, true, visitor) {
                        none => {
                            self.session.span_err(span,
                                                  ~"attempt to implement an \
                                                    unknown trait");
                        }
                        some(def) => {
                            self.record_def(trait_reference.ref_id, def);

                            // Record the current trait reference.
                            (*new_trait_refs).push(def_id_of_def(def));
                        }
                    }
                }

                // Record the current set of trait references.
                self.current_trait_refs = some(new_trait_refs);
            }

            // Resolve the self type.
            self.resolve_type(self_type, visitor);

            for methods.each |method| {
                // We also need a new scope for the method-specific
                // type parameters.
                self.resolve_method(MethodRibKind(id, Provided(method.id)),
                                    method,
                                    outer_type_parameter_count,
                                    visitor);
/*
                let borrowed_type_parameters = &method.tps;
                self.resolve_function(MethodRibKind(id, Provided(method.id)),
                                      some(@method.decl),
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
        debug!{"(resolving module) resolving module ID %d", id};
        self.impl_map.insert(id, self.current_module.impl_scopes);

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
            none => {
                // Nothing to do.
            }
            some(initializer) => {
                self.resolve_expr(initializer.expr, visitor);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern(local.node.pat, IrrefutableMode, mutability,
                             none, visitor);
    }

    fn binding_mode_map(pat: @pat) -> BindingMap {
        let result = box_str_hash();
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
        for arm.pats.eachi() |i, p: @pat| {
            let map_i = self.binding_mode_map(p);

            for map_0.each |key, binding_0| {
                match map_i.find(key) {
                  none => {
                    self.session.span_err(
                        p.span,
                        fmt!{"variable `%s` from pattern #1 is \
                                  not bound in pattern #%u",
                             *key, i + 1});
                  }
                  some(binding_i) => {
                    if binding_0.binding_mode != binding_i.binding_mode {
                        self.session.span_err(
                            binding_i.span,
                            fmt!{"variable `%s` is bound with different \
                                      mode in pattern #%u than in pattern #1",
                                 *key, i + 1});
                    }
                  }
                }
            }

            for map_i.each |key, binding| {
                if !map_0.contains_key(key) {
                    self.session.span_err(
                        binding.span,
                        fmt!{"variable `%s` from pattern #%u is \
                                  not bound in pattern #1",
                             *key, i + 1});
                }
            }
        }
    }

    fn resolve_arm(arm: arm, visitor: ResolveVisitor) {
        (*self.value_ribs).push(@Rib(NormalRibKind));

        let bindings_list = atom_hashmap();
        for arm.pats.each |pattern| {
            self.resolve_pattern(pattern, RefutableMode, Immutable,
                                 some(bindings_list), visitor);
        }

        // This has to happen *after* we determine which
        // pat_idents are variants
        self.check_consistent_bindings(arm);

        visit_expr_opt(arm.guard, (), visitor);
        self.resolve_block(arm.body, visitor);

        (*self.value_ribs).pop();
    }

    fn resolve_block(block: blk, visitor: ResolveVisitor) {
        debug!{"(resolving block) entering block"};
        (*self.value_ribs).push(@Rib(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        match self.current_module.anonymous_children.find(block.node.id) {
            none => { /* Nothing to do. */ }
            some(anonymous_module) => {
                debug!{"(resolving block) found anonymous module, moving \
                        down"};
                self.current_module = anonymous_module;
            }
        }

        // Descend into the block.
        visit_block(block, (), visitor);

        // Move back up.
        self.current_module = orig_module;

        (*self.value_ribs).pop();
        debug!{"(resolving block) leaving block"};
    }

    fn resolve_type(ty: @ty, visitor: ResolveVisitor) {
        match ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            ty_path(path, path_id) => {
                // This is a path in the type namespace. Walk through scopes
                // scopes looking for it.

                let mut result_def;
                match self.resolve_path(path, TypeNS, true, visitor) {
                    some(def) => {
                        debug!{"(resolving type) resolved `%s` to type",
                               *path.idents.last()};
                        result_def = some(def);
                    }
                    none => {
                        result_def = none;
                    }
                }

                match result_def {
                    some(_) => {
                        // Continue.
                    }
                    none => {
                        // Check to see whether the name is a primitive type.
                        if path.idents.len() == 1u {
                            let name =
                                (*self.atom_table).intern(path.idents.last());

                            match self.primitive_type_table
                                    .primitive_types
                                    .find(name) {

                                some(primitive_type) => {
                                    result_def =
                                        some(def_prim_ty(primitive_type));
                                }
                                none => {
                                    // Continue.
                                }
                            }
                        }
                    }
                }

                match copy result_def {
                    some(def) => {
                        // Write the result into the def map.
                        debug!{"(resolving type) writing resolution for `%s` \
                                (id %d)",
                               connect(path.idents.map(|x| *x), ~"::"),
                               path_id};
                        self.record_def(path_id, def);
                    }
                    none => {
                        self.session.span_err
                            (ty.span, fmt!{"use of undeclared type name `%s`",
                                           connect(path.idents.map(|x| *x),
                                                   ~"::")});
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
                       bindings_list: option<hashmap<Atom,node_id>>,
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

                    let atom = (*self.atom_table).intern(path.idents[0]);

                    match self.resolve_enum_variant_or_const(atom) {
                        FoundEnumVariant(def) if mode == RefutableMode => {
                            debug!{"(resolving pattern) resolving `%s` to \
                                    enum variant",
                                   *path.idents[0]};

                            self.record_def(pattern.id, def);
                        }
                        FoundEnumVariant(_) => {
                            self.session.span_err(pattern.span,
                                                  fmt!{"declaration of `%s` \
                                                        shadows an enum \
                                                        that's in scope",
                                                       *(*self.atom_table).
                                                            atom_to_str
                                                            (atom)});
                        }
                        FoundConst => {
                            self.session.span_err(pattern.span,
                                                  ~"pattern variable \
                                                   conflicts with a constant \
                                                   in scope");
                        }
                        EnumVariantOrConstNotFound => {
                            debug!{"(resolving pattern) binding `%s`",
                                   *path.idents[0]};

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
                                some(bindings_list)
                                if !bindings_list.contains_key(atom) => {
                                    let last_rib = (*self.value_ribs).last();
                                    last_rib.bindings.insert(atom,
                                                             dl_def(def));
                                    bindings_list.insert(atom, pat_id);
                                }
                                some(b) => {
                                  if b.find(atom) == some(pat_id) {
                                      // Then this is a duplicate variable
                                      // in the same disjunct, which is an
                                      // error
                                     self.session.span_err(pattern.span,
                                       fmt!{"Identifier %s is bound more \
                                             than once in the same pattern",
                                            path_to_str(path)});
                                  }
                                  // Not bound in the same pattern: do nothing
                                }
                                none => {
                                    let last_rib = (*self.value_ribs).last();
                                    last_rib.bindings.insert(atom,
                                                             dl_def(def));
                                }
                            }
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(ty, visitor);
                    }
                }

                pat_ident(_, path, _) | pat_enum(path, _) => {
                    // These two must be enum variants.
                    match self.resolve_path(path, ValueNS, false, visitor) {
                        some(def @ def_variant(*)) => {
                            self.record_def(pattern.id, def);
                        }
                        some(_) => {
                            self.session.span_err(path.span,
                                                  fmt!{"not an enum \
                                                        variant: %s",
                                                       *path.idents.last()});
                        }
                        none => {
                            self.session.span_err(path.span,
                                                  ~"unresolved enum variant");
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(ty, visitor);
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
                        some(def_ty(class_id))
                                if self.structs.contains_key(class_id) => {
                            let has_constructor = self.structs.get(class_id);
                            let class_def = def_class(class_id,
                                                      has_constructor);
                            self.record_def(pattern.id, class_def);
                        }
                        some(definition @ def_variant(_, variant_id))
                                if self.structs.contains_key(variant_id) => {
                            self.record_def(pattern.id, definition);
                        }
                        _ => {
                            self.session.span_err(path.span,
                                                  fmt!("`%s` does not name a \
                                                        structure",
                                                       connect(path.idents.map
                                                               (|x| *x),
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

    fn resolve_enum_variant_or_const(name: Atom)
                                  -> EnumVariantOrConstResolution {

        match self.resolve_item_in_lexical_scope(self.current_module,
                                               name,
                                               ValueNS) {

            Success(target) => {
                match target.bindings.value_def {
                    none => {
                        fail ~"resolved name in the value namespace to a set \
                              of name bindings with no def?!";
                    }
                    some(def @ def_variant(*)) => {
                        return FoundEnumVariant(def);
                    }
                    some(def_const(*)) => {
                        return FoundConst;
                    }
                    some(_) => {
                        return EnumVariantOrConstNotFound;
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
                 -> option<def> {

        // First, resolve the types.
        for path.types.each |ty| {
            self.resolve_type(ty, visitor);
        }

        if path.global {
            return self.resolve_crate_relative_path(path,
                                                 self.xray_context,
                                                 namespace);
        }

        if path.idents.len() > 1u {
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
                       -> option<def> {

        if check_ribs {
            match self.resolve_identifier_in_local_ribs(identifier,
                                                      namespace,
                                                      span) {
                some(def) => {
                    return some(def);
                }
                none => {
                    // Continue.
                }
            }
        }

        return self.resolve_item_by_identifier_in_lexical_scope(identifier,
                                                             namespace);
    }

    // XXX: Merge me with resolve_name_in_module?
    fn resolve_definition_of_name_in_module(containing_module: @Module,
                                            name: Atom,
                                            namespace: Namespace,
                                            xray: XrayFlag)
                                         -> NameDefinition {

        if xray == NoXray && !self.name_is_exported(containing_module, name) {
            debug!{"(resolving definition of name in module) name `%s` is \
                    unexported",
                   *(*self.atom_table).atom_to_str(name)};
            return NoNameDefinition;
        }

        // First, search children.
        match containing_module.children.find(name) {
            some(child_name_bindings) => {
                match (*child_name_bindings).def_for_namespace(namespace) {
                    some(def) => {
                        // Found it. Stop the search here.
                        return ChildNameDefinition(def);
                    }
                    none => {
                        // Continue.
                    }
                }
            }
            none => {
                // Continue.
            }
        }

        // Next, search import resolutions.
        match containing_module.import_resolutions.find(name) {
            some(import_resolution) => {
                match (*import_resolution).target_for_namespace(namespace) {
                    some(target) => {
                        match (*target.bindings)
                            .def_for_namespace(namespace) {
                            some(def) => {
                                // Found it.
                                import_resolution.used = true;
                                return ImportNameDefinition(def);
                            }
                            none => {
                                // This can happen with external impls, due to
                                // the imperfect way we read the metadata.

                                return NoNameDefinition;
                            }
                        }
                    }
                    none => {
                        return NoNameDefinition;
                    }
                }
            }
            none => {
                return NoNameDefinition;
            }
        }
    }

    fn intern_module_part_of_path(path: @path) -> @dvec<Atom> {
        let module_path_atoms = @dvec();
        for path.idents.eachi |index, ident| {
            if index == path.idents.len() - 1u {
                break;
            }

            (*module_path_atoms).push((*self.atom_table).intern(ident));
        }

        return module_path_atoms;
    }

    fn resolve_module_relative_path(path: @path,
                                    +xray: XrayFlag,
                                    namespace: Namespace)
                                 -> option<def> {

        let module_path_atoms = self.intern_module_part_of_path(path);

        let mut containing_module;
        match self.resolve_module_path_for_import(self.current_module,
                                                module_path_atoms,
                                                xray,
                                                path.span) {

            Failed => {
                self.session.span_err(path.span,
                                      fmt!{"use of undeclared module `%s`",
                                            *(*self.atom_table).atoms_to_str
                                              ((*module_path_atoms).get())});
                return none;
            }

            Indeterminate => {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = (*self.atom_table).intern(path.idents.last());
        match self.resolve_definition_of_name_in_module(containing_module,
                                                      name,
                                                      namespace,
                                                      xray) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                self.session.span_err(path.span,
                                      fmt!{"unresolved name: %s::%s",
                                           *(*self.atom_table).atoms_to_str
                                               ((*module_path_atoms).get()),
                                           *(*self.atom_table).atom_to_str
                                               (name)});
                return none;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) => {
                return some(def);
            }
        }
    }

    fn resolve_crate_relative_path(path: @path,
                                   +xray: XrayFlag,
                                   namespace: Namespace)
                                -> option<def> {

        let module_path_atoms = self.intern_module_part_of_path(path);

        let root_module = (*self.graph_root).get_module();

        let mut containing_module;
        match self.resolve_module_path_from_root(root_module,
                                               module_path_atoms,
                                               0u,
                                               xray,
                                               path.span) {

            Failed => {
                self.session.span_err(path.span,
                                      fmt!{"use of undeclared module `::%s`",
                                            *(*self.atom_table).atoms_to_str
                                              ((*module_path_atoms).get())});
                return none;
            }

            Indeterminate => {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) => {
                containing_module = resulting_module;
            }
        }

        let name = (*self.atom_table).intern(path.idents.last());
        match self.resolve_definition_of_name_in_module(containing_module,
                                                      name,
                                                      namespace,
                                                      xray) {
            NoNameDefinition => {
                // We failed to resolve the name. Report an error.
                self.session.span_err(path.span,
                                      fmt!{"unresolved name: %s::%s",
                                           *(*self.atom_table).atoms_to_str
                                               ((*module_path_atoms).get()),
                                           *(*self.atom_table).atom_to_str
                                               (name)});
                return none;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) => {
                return some(def);
            }
        }
    }

    fn resolve_identifier_in_local_ribs(identifier: ident,
                                        namespace: Namespace,
                                        span: span)
                                     -> option<def> {

        let name = (*self.atom_table).intern(identifier);

        // Check the local set of ribs.
        let mut search_result;
        match namespace {
            ValueNS => {
                search_result = self.search_ribs(self.value_ribs, name, span,
                                                 DontAllowCapturingSelf);
            }
            TypeNS => {
                search_result = self.search_ribs(self.type_ribs, name, span,
                                                 AllowCapturingSelf);
            }
            ModuleNS | ImplNS => {
                fail ~"module or impl namespaces do not have local ribs";
            }
        }

        match copy search_result {
            some(dl_def(def)) => {
                debug!{"(resolving path in local ribs) resolved `%s` to \
                        local: %?",
                       *(*self.atom_table).atom_to_str(name),
                       def};
                return some(def);
            }
            some(dl_field) | some(dl_impl(_)) | none => {
                return none;
            }
        }
    }

    fn resolve_item_by_identifier_in_lexical_scope(ident: ident,
                                                   namespace: Namespace)
                                                -> option<def> {

        let name = (*self.atom_table).intern(ident);

        // Check the items.
        match self.resolve_item_in_lexical_scope(self.current_module,
                                               name,
                                               namespace) {

            Success(target) => {
                match (*target.bindings).def_for_namespace(namespace) {
                    none => {
                        fail ~"resolved name in a namespace to a set of name \
                              bindings with no def for that namespace?!";
                    }
                    some(def) => {
                        debug!{"(resolving item path in lexical scope) \
                                resolved `%s` to item",
                               *(*self.atom_table).atom_to_str(name)};
                        return some(def);
                    }
                }
            }
            Indeterminate => {
                fail ~"unexpected indeterminate result";
            }
            Failed => {
                return none;
            }
        }
    }

    fn resolve_expr(expr: @expr, visitor: ResolveVisitor) {
        // First, write the implementations in scope into a table if the
        // expression might need them.

        self.record_impls_for_expr_if_necessary(expr);

        // Then record candidate traits for this expression if it could result
        // in the invocation of a method call.

        self.record_candidate_traits_for_expr_if_necessary(expr);

        // Next, resolve the node.
        match expr.node {
            // The interpretation of paths depends on whether the path has
            // multiple elements in it or not.

            expr_path(path) => {
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.

                match self.resolve_path(path, ValueNS, true, visitor) {
                    some(def) => {
                        // Write the result into the def map.
                        debug!{"(resolving expr) resolved `%s`",
                               connect(path.idents.map(|x| *x), ~"::")};
                        self.record_def(expr.id, def);
                    }
                    none => {
                        self.session.span_err(expr.span,
                                              fmt!{"unresolved name: %s",
                                              connect(path.idents.map(|x| *x),
                                                      ~"::")});
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_fn(_, fn_decl, block, capture_clause) |
            expr_fn_block(fn_decl, block, capture_clause) => {
                self.resolve_function(FunctionRibKind(expr.id),
                                      some(@fn_decl),
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
                    some(def_ty(class_id))
                            if self.structs.contains_key(class_id) => {
                        let has_constructor = self.structs.get(class_id);
                        let class_def = def_class(class_id, has_constructor);
                        self.record_def(expr.id, class_def);
                    }
                    some(definition @ def_variant(_, class_id))
                            if self.structs.contains_key(class_id) => {
                        self.record_def(expr.id, definition);
                    }
                    _ => {
                        self.session.span_err(path.span,
                                              fmt!{"`%s` does not name a \
                                                    structure",
                                                   connect(path.idents.map
                                                           (|x| *x),
                                                           ~"::")});
                    }
                }

                visit_expr(expr, (), visitor);
            }

            _ => {
                visit_expr(expr, (), visitor);
            }
        }
    }

    fn record_impls_for_expr_if_necessary(expr: @expr) {
        match expr.node {
            expr_field(*) | expr_path(*) | expr_cast(*) | expr_binary(*) |
            expr_unary(*) | expr_assign_op(*) | expr_index(*) => {
                self.impl_map.insert(expr.id,
                                     self.current_module.impl_scopes);
            }
            _ => {
                // Nothing to do.
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(expr: @expr) {
        match expr.node {
            expr_field(_, ident, _) => {
                let atom = (*self.atom_table).intern(ident);
                let traits = self.search_for_traits_containing_method(atom);
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

    fn search_for_traits_containing_method(name: Atom) -> @dvec<def_id> {
        let found_traits = @dvec();
        let mut search_module = self.current_module;
        loop {
            // Look for the current trait.
            match copy self.current_trait_refs {
                some(trait_def_ids) => {
                    for trait_def_ids.each |trait_def_id| {
                        self.add_trait_info_if_containing_method
                            (found_traits, trait_def_id, name);
                    }
                }
                none => {
                    // Nothing to do.
                }
            }

            // Look for trait children.
            for search_module.children.each |_name, child_name_bindings| {
                match child_name_bindings.def_for_namespace(TypeNS) {
                    some(def_ty(trait_def_id)) => {
                        self.add_trait_info_if_containing_method(found_traits,
                                                                 trait_def_id,
                                                                 name);
                    }
                    some(_) | none => {
                        // Continue.
                    }
                }
            }

            // Look for imports.
            for search_module.import_resolutions.each
                    |_atom, import_resolution| {

                match import_resolution.target_for_namespace(TypeNS) {
                    none => {
                        // Continue.
                    }
                    some(target) => {
                        match target.bindings.def_for_namespace(TypeNS) {
                            some(def_ty(trait_def_id)) => {
                                self.add_trait_info_if_containing_method
                                    (found_traits, trait_def_id, name);
                            }
                            some(_) | none => {
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

    fn add_trait_info_if_containing_method(found_traits: @dvec<def_id>,
                                           trait_def_id: def_id,
                                           name: Atom) {

        match self.trait_info.find(trait_def_id) {
            some(trait_info) if trait_info.contains_key(name) => {
                debug!{"(adding trait info if containing method) found trait \
                        %d:%d for method '%s'",
                       trait_def_id.crate,
                       trait_def_id.node,
                       *(*self.atom_table).atom_to_str(name)};
                (*found_traits).push(trait_def_id);
            }
            some(_) | none => {
                // Continue.
            }
        }
    }

    fn add_fixed_trait_for_expr(expr_id: node_id, +trait_id: option<def_id>) {
        let traits = @dvec();
        traits.push(trait_id.get());
        self.trait_map.insert(expr_id, traits);
    }

    fn record_def(node_id: node_id, def: def) {
        debug!{"(recording def) recording %? for %?", def, node_id};
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
            some(def_id) if def_id.crate == local_crate => {
                // OK. Continue.
            }
            none => {
                // Check for unused imports in the root module.
            }
            some(_) => {
                // Bail out.
                debug!{"(checking for unused imports in module subtree) not \
                        checking for unused imports for `%s`",
                       self.module_to_str(module_)};
                return;
            }
        }

        self.check_for_unused_imports_in_module(module_);

        for module_.children.each |_atom, child_name_bindings| {
            match (*child_name_bindings).get_module_if_available() {
                none => {
                    // Nothing to do.
                }
                some(child_module) => {
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
        for module_.import_resolutions.each |_impl_name, import_resolution| {
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
        let atoms = dvec();
        let mut current_module = module_;
        loop {
            match current_module.parent_link {
                NoParentLink => {
                    break;
                }
                ModuleParentLink(module_, name) => {
                    atoms.push(name);
                    current_module = module_;
                }
                BlockParentLink(module_, node_id) => {
                    atoms.push((*self.atom_table).intern(@~"<opaque>"));
                    current_module = module_;
                }
            }
        }

        if atoms.len() == 0u {
            return ~"???";
        }

        let mut string = ~"";
        let mut i = atoms.len() - 1u;
        loop {
            if i < atoms.len() - 1u {
                string += ~"::";
            }
            string += *(*self.atom_table).atom_to_str(atoms.get_elt(i));

            if i == 0u {
                break;
            }
            i -= 1u;
        }

        return string;
    }

    fn dump_module(module_: @Module) {
        debug!{"Dump of module `%s`:", self.module_to_str(module_)};

        debug!{"Children:"};
        for module_.children.each |name, _child| {
            debug!{"* %s", *(*self.atom_table).atom_to_str(name)};
        }

        debug!{"Import resolutions:"};
        for module_.import_resolutions.each |name, import_resolution| {
            let mut module_repr;
            match (*import_resolution).target_for_namespace(ModuleNS) {
                none => { module_repr = ~""; }
                some(target) => {
                    module_repr = ~" module:?";
                    // XXX
                }
            }

            let mut value_repr;
            match (*import_resolution).target_for_namespace(ValueNS) {
                none => { value_repr = ~""; }
                some(target) => {
                    value_repr = ~" value:?";
                    // XXX
                }
            }

            let mut type_repr;
            match (*import_resolution).target_for_namespace(TypeNS) {
                none => { type_repr = ~""; }
                some(target) => {
                    type_repr = ~" type:?";
                    // XXX
                }
            }

            let mut impl_repr;
            match (*import_resolution).target_for_namespace(ImplNS) {
                none => { impl_repr = ~""; }
                some(target) => {
                    impl_repr = ~" impl:?";
                    // XXX
                }
            }

            debug!{"* %s:%s%s%s%s",
                   *(*self.atom_table).atom_to_str(name),
                   module_repr, value_repr, type_repr, impl_repr};
        }
    }

    fn dump_impl_scopes(impl_scopes: ImplScopes) {
        debug!{"Dump of impl scopes:"};

        let mut i = 0u;
        let mut impl_scopes = impl_scopes;
        loop {
            match *impl_scopes {
                cons(impl_scope, rest_impl_scopes) => {
                    debug!{"Impl scope %u:", i};

                    for (*impl_scope).each |implementation| {
                        debug!{"Impl: %s", *implementation.ident};
                    }

                    i += 1u;
                    impl_scopes = rest_impl_scopes;
                }
                nil => {
                    break;
                }
            }
        }
    }
}

/// Entry point to crate resolution.
fn resolve_crate(session: session, lang_items: LanguageItems, crate: @crate)
              -> { def_map: DefMap,
                   exp_map: ExportMap,
                   impl_map: ImplMap,
                   trait_map: TraitMap } {

    let resolver = @Resolver(session, lang_items, crate);
    (*resolver).resolve(resolver);
    return {
        def_map: resolver.def_map,
        exp_map: resolver.export_map,
        impl_map: resolver.impl_map,
        trait_map: resolver.trait_map
    };
}

