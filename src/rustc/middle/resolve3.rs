import driver::session::session;
import metadata::csearch::{each_path, get_impls_for_mod};
import metadata::csearch::{get_method_names_if_trait, lookup_defs};
import metadata::cstore::find_use_stmt_cnum;
import metadata::decoder::{def_like, dl_def, dl_field, dl_impl};
import middle::lint::{error, ignore, level, unused_imports, warn};
import syntax::ast::{_mod, arm, blk, bound_const, bound_copy, bound_trait};
import syntax::ast::{bound_owned};
import syntax::ast::{bound_send, capture_clause, class_ctor, class_dtor};
import syntax::ast::{class_member, class_method, crate, crate_num, decl_item};
import syntax::ast::{def, def_arg, def_binding, def_class, def_const, def_fn};
import syntax::ast::{def_foreign_mod, def_id, def_local, def_mod};
import syntax::ast::{def_prim_ty, def_region, def_self, def_ty, def_ty_param};
import syntax::ast::{def_upvar, def_use, def_variant, expr, expr_assign_op};
import syntax::ast::{expr_binary, expr_cast, expr_field, expr_fn};
import syntax::ast::{expr_fn_block, expr_index, expr_new, expr_path};
import syntax::ast::{expr_unary, fn_decl, foreign_item, foreign_item_fn};
import syntax::ast::{ident, trait_ref, impure_fn, instance_var, item};
import syntax::ast::{item_class, item_const, item_enum, item_fn, item_mac};
import syntax::ast::{item_foreign_mod, item_trait, item_impl, item_mod};
import syntax::ast::{item_ty, local, local_crate, method, node_id, pat};
import syntax::ast::{pat_enum, pat_ident, path, prim_ty, stmt_decl, ty,
                     pat_box, pat_uniq, pat_lit, pat_range, pat_rec,
                     pat_tup, pat_wild};
import syntax::ast::{ty_bool, ty_char, ty_f, ty_f32, ty_f64};
import syntax::ast::{ty_float, ty_i, ty_i16, ty_i32, ty_i64, ty_i8, ty_int};
import syntax::ast::{ty_param, ty_path, ty_str, ty_u, ty_u16, ty_u32, ty_u64};
import syntax::ast::{ty_u8, ty_uint, variant, view_item, view_item_export};
import syntax::ast::{view_item_import, view_item_use, view_path_glob};
import syntax::ast::{view_path_list, view_path_simple};
import syntax::ast::{required, provided};
import syntax::ast_util::{def_id_of_def, dummy_sp, local_def, new_def_hash};
import syntax::ast_util::{walk_pat};
import syntax::attr::{attr_metas, contains_name};
import syntax::print::pprust::path_to_str;
import syntax::codemap::span;
import syntax::visit::{default_visitor, fk_method, mk_vt, visit_block};
import syntax::visit::{visit_crate, visit_expr, visit_expr_opt, visit_fn};
import syntax::visit::{visit_foreign_item, visit_item, visit_method_helper};
import syntax::visit::{visit_mod, visit_ty, vt};

import box::ptr_eq;
import dvec::{dvec, extensions};
import option::get;
import str::{connect, split_str};
import vec::pop;

import std::list::{cons, list, nil};
import std::map::{hashmap, int_hash, str_hash};
import ASTMap = syntax::ast_map::map;
import str_eq = str::eq;

// Definition mapping
type DefMap = hashmap<node_id,def>;

// Implementation resolution
type MethodInfo = { did: def_id, n_tps: uint, ident: ident };
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

    // We passed through a function *item* scope. Disallow upvars.
    OpaqueFunctionRibKind
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
    ret n;
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
        alt self.atoms.find(string) {
            none { /* fall through */ }
            some(atom) { ret atom; }
        }

        let atom = Atom(self.atom_count);
        self.atom_count += 1u;
        self.atoms.insert(string, atom);
        self.strings.push(string);

        ret atom;
    }

    fn atom_to_str(atom: Atom) -> @~str {
        ret self.strings.get_elt(atom);
    }

    fn atoms_to_strs(atoms: ~[Atom], f: fn(@~str) -> bool) {
        for atoms.each |atom| {
            if !f(self.atom_to_str(atom)) {
                ret;
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
        ret @result;
    }
}

/// Creates a hash table of atoms.
fn atom_hashmap<V:copy>() -> hashmap<Atom,V> {
    ret hashmap::<Atom,V>(|a| a, |a, b| a == b);
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
        alt namespace {
            ModuleNS    { ret copy self.module_target; }
            TypeNS      { ret copy self.type_target;   }
            ValueNS     { ret copy self.value_target;  }

            ImplNS {
                if (*self.impl_target).len() > 0u {
                    ret some(copy *(*self.impl_target).get_elt(0u));
                }
                ret none;
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
        ret self.imports.len() == self.resolved_import_count;
    }
}

// XXX: This is a workaround due to is_none in the standard library mistakenly
// requiring a T:copy.

pure fn is_none<T>(x: option<T>) -> bool {
    alt x {
        none { ret true; }
        some(_) { ret false; }
    }
}

fn unused_import_lint_level(session: session) -> level {
    for session.opts.lint_opts.each |lint_option_pair| {
        let (lint_type, lint_level) = lint_option_pair;
        if lint_type == unused_imports {
            ret lint_level;
        }
    }
    ret ignore;
}

// Records the definitions (at most one for each namespace) that a name is
// bound to.
class NameBindings {
    let mut module_def: ModuleDef;      //< Meaning in the module namespace.
    let mut type_def: option<def>;      //< Meaning in the type namespace.
    let mut value_def: option<def>;     //< Meaning in the value namespace.
    let mut impl_defs: ~[@Impl];        //< Meaning in the impl namespace.

    new() {
        self.module_def = NoModuleDef;
        self.type_def = none;
        self.value_def = none;
        self.impl_defs = ~[];
    }

    /// Creates a new module in this set of name bindings.
    fn define_module(parent_link: ParentLink, def_id: option<def_id>) {
        if self.module_def == NoModuleDef {
            let module = @Module(parent_link, def_id);
            self.module_def = ModuleDef(module);
        }
    }

    /// Records a type definition.
    fn define_type(def: def) {
        self.type_def = some(def);
    }

    /// Records a value definition.
    fn define_value(def: def) {
        self.value_def = some(def);
    }

    /// Records an impl definition.
    fn define_impl(implementation: @Impl) {
        self.impl_defs += ~[implementation];
    }

    /// Returns the module node if applicable.
    fn get_module_if_available() -> option<@Module> {
        alt self.module_def {
            NoModuleDef         { ret none;         }
            ModuleDef(module)   { ret some(module); }
        }
    }

    /**
     * Returns the module node. Fails if this node does not have a module
     * definition.
     */
    fn get_module() -> @Module {
        alt self.module_def {
            NoModuleDef {
                fail
                    ~"get_module called on a node with no module definition!";
            }
            ModuleDef(module) {
                ret module;
            }
        }
    }

    fn defined_in_namespace(namespace: Namespace) -> bool {
        alt namespace {
            ModuleNS    { ret self.module_def != NoModuleDef; }
            TypeNS      { ret self.type_def != none;          }
            ValueNS     { ret self.value_def != none;         }
            ImplNS      { ret self.impl_defs.len() >= 1u;     }
        }
    }

    fn def_for_namespace(namespace: Namespace) -> option<def> {
        alt namespace {
            TypeNS {
                ret self.type_def;
            }
            ValueNS {
                ret self.value_def;
            }
            ModuleNS {
                alt self.module_def {
                    NoModuleDef {
                        ret none;
                    }
                    ModuleDef(module) {
                        alt module.def_id {
                            none {
                                ret none;
                            }
                            some(def_id) {
                                ret some(def_mod(def_id));
                            }
                        }
                    }
                }
            }
            ImplNS {
                // Danger: Be careful what you use this for! def_ty is not
                // necessarily the right def.

                if self.impl_defs.len() == 0u {
                    ret none;
                }
                ret some(def_ty(self.impl_defs[0].did));
            }
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

/// The main resolver class.
class Resolver {
    let session: session;
    let ast_map: ASTMap;
    let crate: @crate;

    let atom_table: @AtomTable;

    let graph_root: @NameBindings;

    let unused_import_lint_level: level;

    let trait_info: hashmap<def_id,@hashmap<Atom,()>>;

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

    new(session: session, ast_map: ASTMap, crate: @crate) {
        self.session = session;
        self.ast_map = ast_map;
        self.crate = crate;

        self.atom_table = @AtomTable();

        // The outermost module has def ID 0; this is not reflected in the
        // AST.

        self.graph_root = @NameBindings();
        (*self.graph_root).define_module(NoParentLink,
                                         some({ crate: 0, node: 0 }));

        self.unused_import_lint_level = unused_import_lint_level(session);

        self.trait_info = new_def_hash();

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
        alt reduced_graph_parent {
            ModuleReducedGraphParent(module) {
                ret module;
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
                 reduced_graph_parent: ReducedGraphParent)
              -> (@NameBindings, ReducedGraphParent) {

        // If this is the immediate descendant of a module, then we add the
        // child name directly. Otherwise, we create or reuse an anonymous
        // module and add the child to that.

        let mut module;
        alt reduced_graph_parent {
            ModuleReducedGraphParent(parent_module) {
                module = parent_module;
            }
        }

        // Add or reuse the child.
        let new_parent = ModuleReducedGraphParent(module);
        alt module.children.find(name) {
            none {
                let child = @NameBindings();
                module.children.insert(name, child);
                ret (child, new_parent);
            }
            some(child) {
                ret (child, new_parent);
            }
        }
    }

    fn block_needs_anonymous_module(block: blk) -> bool {
        // If the block has view items, we need an anonymous module.
        if block.node.view_items.len() > 0u {
            ret true;
        }

        // Check each statement.
        for block.node.stmts.each |statement| {
            alt statement.node {
                stmt_decl(declaration, _) {
                    alt declaration.node {
                        decl_item(_) {
                            ret true;
                        }
                        _ {
                            // Keep searching.
                        }
                    }
                }
                _ {
                    // Keep searching.
                }
            }
        }

        // If we found neither view items nor items, we don't need to create
        // an anonymous module.

        ret false;
    }

    fn get_parent_link(parent: ReducedGraphParent, name: Atom) -> ParentLink {
        alt parent {
            ModuleReducedGraphParent(module) {
                ret ModuleParentLink(module, name);
            }
        }
    }

    /// Constructs the reduced graph for one item.
    fn build_reduced_graph_for_item(item: @item,
                                    parent: ReducedGraphParent,
                                    &&visitor: vt<ReducedGraphParent>) {

        let atom = (*self.atom_table).intern(item.ident);
        let (name_bindings, new_parent) = self.add_child(atom, parent);

        alt item.node {
            item_mod(module) {
                let parent_link = self.get_parent_link(new_parent, atom);
                let def_id = { crate: 0, node: item.id };
                (*name_bindings).define_module(parent_link, some(def_id));

                let new_parent =
                    ModuleReducedGraphParent((*name_bindings).get_module());

                visit_mod(module, item.span, item.id, new_parent, visitor);
            }
            item_foreign_mod(foreign_module) {
                let parent_link = self.get_parent_link(new_parent, atom);
                let def_id = { crate: 0, node: item.id };
                (*name_bindings).define_module(parent_link, some(def_id));

                let new_parent =
                    ModuleReducedGraphParent((*name_bindings).get_module());

                visit_item(item, new_parent, visitor);
            }

            // These items live in the value namespace.
            item_const(*) {
                (*name_bindings).define_value(def_const(local_def(item.id)));
            }
            item_fn(decl, _, _) {
                let def = def_fn(local_def(item.id), decl.purity);
                (*name_bindings).define_value(def);
                visit_item(item, new_parent, visitor);
            }

            // These items live in the type namespace.
            item_ty(*) {
                (*name_bindings).define_type(def_ty(local_def(item.id)));
            }

            // These items live in both the type and value namespaces.
            item_enum(variants, _) {
                (*name_bindings).define_type(def_ty(local_def(item.id)));

                for variants.each |variant| {
                    self.build_reduced_graph_for_variant(variant,
                                                         local_def(item.id),
                                                         new_parent,
                                                         visitor);
                }
            }
            item_class(_, _, class_members, ctor, _) {
                (*name_bindings).define_type(def_ty(local_def(item.id)));

                let purity = ctor.node.dec.purity;
                let ctor_def = def_fn(local_def(ctor.node.id), purity);
                (*name_bindings).define_value(ctor_def);

                // Create the set of implementation information that the
                // implementation scopes (ImplScopes) need and write it into
                // the implementation definition list for this set of name
                // bindings.

                let mut method_infos = ~[];
                for class_members.each |class_member| {
                    alt class_member.node {
                        class_method(method) {
                            // XXX: Combine with impl method code below.
                            method_infos += ~[
                                @{
                                    did: local_def(method.id),
                                    n_tps: method.tps.len(),
                                    ident: method.ident
                                }
                            ];
                        }
                        instance_var(*) {
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

                visit_item(item, new_parent, visitor);
            }

            item_impl(_, _, _, methods) {
                // Create the set of implementation information that the
                // implementation scopes (ImplScopes) need and write it into
                // the implementation definition list for this set of name
                // bindings.

                let mut method_infos = ~[];
                for methods.each |method| {
                    method_infos += ~[
                        @{
                            did: local_def(method.id),
                            n_tps: method.tps.len(),
                            ident: method.ident
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

            item_trait(_, methods) {
                // Add the names of all the methods to the trait info.
                let method_names = @atom_hashmap();
                for methods.each |method| {
                    let atom;
                    alt method {
                        required(required_method) {
                            atom = (*self.atom_table).intern
                                (required_method.ident);
                        }
                        provided(provided_method) {
                            atom = (*self.atom_table).intern
                                (provided_method.ident);
                        }
                    }
                    (*method_names).insert(atom, ());
                }

                let def_id = local_def(item.id);
                self.trait_info.insert(def_id, method_names);

                (*name_bindings).define_type(def_ty(def_id));
                visit_item(item, new_parent, visitor);
            }

            item_mac(*) {
                fail ~"item macros unimplemented"
            }
        }
    }

    /**
     * Constructs the reduced graph for one variant. Variants exist in the
     * type namespace.
     */
    fn build_reduced_graph_for_variant(variant: variant,
                                       item_id: def_id,
                                       parent: ReducedGraphParent,
                                       &&_visitor: vt<ReducedGraphParent>) {

        let atom = (*self.atom_table).intern(variant.node.name);
        let (child, _) = self.add_child(atom, parent);

        (*child).define_value(def_variant(item_id,
                                          local_def(variant.node.id)));
    }

    /**
     * Constructs the reduced graph for one 'view item'. View items consist
     * of imports and use directives.
     */
    fn build_reduced_graph_for_view_item(view_item: @view_item,
                                         parent: ReducedGraphParent,
                                         &&_visitor: vt<ReducedGraphParent>) {
        alt view_item.node {
            view_item_import(view_paths) {
                for view_paths.each |view_path| {
                    // Extract and intern the module part of the path. For
                    // globs and lists, the path is found directly in the AST;
                    // for simple paths we have to munge the path a little.

                    let module_path = @dvec();
                    alt view_path.node {
                        view_path_simple(_, full_path, _) {
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
                        view_path_list(module_ident_path, _, _) {
                            for module_ident_path.idents.each |ident| {
                                let atom = (*self.atom_table).intern(ident);
                                (*module_path).push(atom);
                            }
                        }
                    }

                    // Build up the import directives.
                    let module = self.get_module_from_parent(parent);
                    alt view_path.node {
                        view_path_simple(binding, full_path, _) {
                            let target_atom =
                                (*self.atom_table).intern(binding);
                            let source_ident = full_path.idents.last();
                            let source_atom =
                                (*self.atom_table).intern(source_ident);
                            let subclass = @SingleImport(target_atom,
                                                         source_atom);
                            self.build_import_directive(module,
                                                        module_path,
                                                        subclass,
                                                        view_path.span);
                        }
                        view_path_list(_, source_idents, _) {
                            for source_idents.each |source_ident| {
                                let name = source_ident.node.name;
                                let atom = (*self.atom_table).intern(name);
                                let subclass = @SingleImport(atom, atom);
                                self.build_import_directive(module,
                                                            module_path,
                                                            subclass,
                                                            view_path.span);
                            }
                        }
                        view_path_glob(_, _) {
                            self.build_import_directive(module,
                                                        module_path,
                                                        @GlobImport,
                                                        view_path.span);
                        }
                    }
                }
            }

            view_item_export(view_paths) {
                let module = self.get_module_from_parent(parent);
                for view_paths.each |view_path| {
                    alt view_path.node {
                        view_path_simple(ident, full_path, ident_id) {
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
                            module.exported_names.insert(atom, ident_id);
                        }

                        view_path_glob(*) {
                            self.session.span_err(view_item.span,
                                                  ~"export globs are \
                                                   unsupported");
                        }

                        view_path_list(path, path_list_idents, _) {
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
                                    module.exported_names.insert(atom, id);
                                }
                            }
                        }
                    }
                }
            }

            view_item_use(name, _, node_id) {
                alt find_use_stmt_cnum(self.session.cstore, node_id) {
                    some(crate_id) {
                        let atom = (*self.atom_table).intern(name);
                        let (child_name_bindings, new_parent) =
                            self.add_child(atom, parent);

                        let def_id = { crate: crate_id, node: 0 };
                        let parent_link = ModuleParentLink
                            (self.get_module_from_parent(new_parent), atom);

                        (*child_name_bindings).define_module(parent_link,
                                                             some(def_id));
                        self.build_reduced_graph_for_external_crate
                            ((*child_name_bindings).get_module());
                    }
                    none {
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
        let (name_bindings, new_parent) = self.add_child(name, parent);

        alt foreign_item.node {
            foreign_item_fn(fn_decl, type_parameters) {
                let def = def_fn(local_def(foreign_item.id), fn_decl.purity);
                (*name_bindings).define_value(def);

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

            #debug("(building reduced graph for block) creating a new \
                    anonymous module for block %d",
                   block_id);

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

    /**
     * Builds the reduced graph rooted at the 'use' directive for an external
     * crate.
     */
    fn build_reduced_graph_for_external_crate(root: @Module) {
        let modules = new_def_hash();

        // Create all the items reachable by paths.
        for each_path(self.session.cstore, get(root.def_id).crate)
                |path_entry| {

            #debug("(building reduced graph for external crate) found path \
                    entry: %s (%?)",
                   path_entry.path_string,
                   path_entry.def_like);

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
                                   ModuleReducedGraphParent(current_module));

                // Define or reuse the module node.
                alt child_name_bindings.module_def {
                    NoModuleDef {
                        #debug("(building reduced graph for external crate) \
                                autovivifying %s", ident);
                        let parent_link = self.get_parent_link(new_parent,
                                                               atom);
                        (*child_name_bindings).define_module(parent_link,
                                                             none);
                    }
                    ModuleDef(_) { /* Fall through. */ }
                }

                current_module = (*child_name_bindings).get_module();
            }

            // Add the new child item.
            let atom = (*self.atom_table).intern(@copy final_ident);
            let (child_name_bindings, new_parent) =
                self.add_child(atom,
                               ModuleReducedGraphParent(current_module));

            alt path_entry.def_like {
                dl_def(def) {
                    alt def {
                        def_mod(def_id) | def_foreign_mod(def_id) {
                            alt copy child_name_bindings.module_def {
                                NoModuleDef {
                                    #debug("(building reduced graph for \
                                            external crate) building module \
                                            %s", final_ident);
                                    let parent_link =
                                        self.get_parent_link(new_parent,
                                                             atom);

                                    alt modules.find(def_id) {
                                        none {
                                            (*child_name_bindings).
                                                define_module(parent_link,
                                                              some(def_id));
                                            modules.insert(def_id,
                                                (*child_name_bindings).
                                                    get_module());
                                        }
                                        some(existing_module) {
                                            // Create an import resolution to
                                            // avoid creating cycles in the
                                            // module graph.

                                            let resolution =
                                                @ImportResolution(dummy_sp());
                                            resolution.
                                                outstanding_references = 0;

                                            alt existing_module.parent_link {
                                                NoParentLink |
                                                        BlockParentLink(*) {
                                                    fail ~"can't happen";
                                                }
                                                ModuleParentLink
                                                        (parent_module,
                                                         atom) {

                                                    let name_bindings =
                                                        parent_module.
                                                            children.get
                                                                (atom);

                                                    resolution.module_target =
                                                        some(Target
                                                            (parent_module,
                                                             name_bindings));
                                                }
                                            }

                                            #debug("(building reduced graph \
                                                     for external crate) \
                                                     ... creating import \
                                                     resolution");

                                            new_parent.import_resolutions.
                                                insert(atom, resolution);
                                        }
                                    }
                                }
                                ModuleDef(module) {
                                    #debug("(building reduced graph for \
                                            external crate) already created \
                                            module");
                                    module.def_id = some(def_id);
                                    modules.insert(def_id, module);
                                }
                            }
                        }
                        def_fn(def_id, _) | def_const(def_id) |
                        def_variant(_, def_id) {
                            #debug("(building reduced graph for external \
                                    crate) building value %s", final_ident);
                            (*child_name_bindings).define_value(def);
                        }
                        def_ty(def_id) {
                            #debug("(building reduced graph for external \
                                    crate) building type %s", final_ident);

                            // If this is a trait, add all the method names
                            // to the trait info.

                            alt get_method_names_if_trait(self.session.cstore,
                                                          def_id) {
                                none {
                                    // Nothing to do.
                                }
                                some(method_names) {
                                    let interned_method_names =
                                        @atom_hashmap();
                                    for method_names.each |method_name| {
                                        #debug("(building reduced graph for \
                                                 external crate) ... adding \
                                                 trait method '%?'",
                                               method_name);
                                        let atom =
                                            (*self.atom_table).intern
                                                (method_name);
                                        (*interned_method_names).insert(atom,
                                                                        ());
                                    }
                                    self.trait_info.insert
                                        (def_id, interned_method_names);
                                }
                            }

                            (*child_name_bindings).define_type(def);
                        }
                        def_class(def_id) {
                            #debug("(building reduced graph for external \
                                    crate) building value and type %s",
                                    final_ident);
                            (*child_name_bindings).define_value(def);
                            (*child_name_bindings).define_type(def);
                        }
                        def_self(*) | def_arg(*) | def_local(*) |
                        def_prim_ty(*) | def_ty_param(*) | def_binding(*) |
                        def_use(*) | def_upvar(*) | def_region(*) {
                            fail #fmt("didn't expect `%?`", def);
                        }
                    }
                }
                dl_impl(_) {
                    // Because of the infelicitous way the metadata is
                    // written, we can't process this impl now. We'll get it
                    // later.

                    #debug("(building reduced graph for external crate) \
                            ignoring impl %s", final_ident);
                }
                dl_field {
                    #debug("(building reduced graph for external crate) \
                            ignoring field %s", final_ident);
                }
            }
        }

        // Create nodes for all the impls.
        self.build_reduced_graph_for_impls_in_external_module_subtree(root);
    }

    fn build_reduced_graph_for_impls_in_external_module_subtree(module:
                                                                @Module) {
        self.build_reduced_graph_for_impls_in_external_module(module);

        for module.children.each |_name, child_node| {
            alt (*child_node).get_module_if_available() {
                none {
                    // Nothing to do.
                }
                some(child_module) {
                    self.
                    build_reduced_graph_for_impls_in_external_module_subtree
                        (child_module);
                }
            }
        }
    }

    fn build_reduced_graph_for_impls_in_external_module(module: @Module) {
        // XXX: This is really unfortunate. decoder::each_path can produce
        // false positives, since, in the crate metadata, a trait named 'bar'
        // in module 'foo' defining a method named 'baz' will result in the
        // creation of a (bogus) path entry named 'foo::bar::baz', and we will
        // create a module node for "bar". We can identify these fake modules
        // by the fact that they have no def ID, which we do here in order to
        // skip them.

        #debug("(building reduced graph for impls in external crate) looking \
                for impls in `%s` (%?)",
               self.module_to_str(module),
               copy module.def_id);

        alt module.def_id {
            none {
                #debug("(building reduced graph for impls in external \
                        module) no def ID for `%s`, skipping",
                       self.module_to_str(module));
                ret;
            }
            some(_) {
                // Continue.
            }
        }

        let impls_in_module = get_impls_for_mod(self.session.cstore,
                                                get(module.def_id),
                                                none);

        // Intern def IDs to prevent duplicates.
        let def_ids = new_def_hash();

        for (*impls_in_module).each |implementation| {
            if def_ids.contains_key(implementation.did) {
                again;
            }
            def_ids.insert(implementation.did, ());

            #debug("(building reduced graph for impls in external module) \
                    added impl `%s` (%?) to `%s`",
                   *implementation.ident,
                   implementation.did,
                   self.module_to_str(module));

            let name = (*self.atom_table).intern(implementation.ident);

            let (name_bindings, _) =
                self.add_child(name, ModuleReducedGraphParent(module));

            name_bindings.impl_defs += ~[implementation];
        }
    }

    /// Creates and adds an import directive to the given module.
    fn build_import_directive(module: @Module,
                              module_path: @dvec<Atom>,
                              subclass: @ImportDirectiveSubclass,
                              span: span) {

        let directive = @ImportDirective(module_path, subclass, span);
        module.imports.push(directive);

        // Bump the reference count on the name. Or, if this is a glob, set
        // the appropriate flag.

        alt *subclass {
            SingleImport(target, _) {
                alt module.import_resolutions.find(target) {
                    some(resolution) {
                        resolution.outstanding_references += 1u;
                    }
                    none {
                        let resolution = @ImportResolution(span);
                        resolution.outstanding_references = 1u;
                        module.import_resolutions.insert(target, resolution);
                    }
                }
            }
            GlobImport {
                // Set the glob flag. This tells us that we don't know the
                // module's exports ahead of time.

                module.glob_count += 1u;
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
            #debug("(resolving imports) iteration %u, %u imports left",
                   i, self.unresolved_imports);

            let module_root = (*self.graph_root).get_module();
            self.resolve_imports_for_module_subtree(module_root);

            if self.unresolved_imports == 0u {
                #debug("(resolving imports) success");
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
    fn resolve_imports_for_module_subtree(module: @Module) {
        #debug("(resolving imports for module subtree) resolving %s",
               self.module_to_str(module));
        self.resolve_imports_for_module(module);

        for module.children.each |_name, child_node| {
            alt (*child_node).get_module_if_available() {
                none {
                    // Nothing to do.
                }
                some(child_module) {
                    self.resolve_imports_for_module_subtree(child_module);
                }
            }
        }

        for module.anonymous_children.each |_block_id, child_module| {
            self.resolve_imports_for_module_subtree(child_module);
        }
    }

    /// Attempts to resolve imports for the given module only.
    fn resolve_imports_for_module(module: @Module) {
        if (*module).all_imports_resolved() {
            #debug("(resolving imports for module) all imports resolved for \
                   %s",
                   self.module_to_str(module));
            ret;
        }

        let import_count = module.imports.len();
        while module.resolved_import_count < import_count {
            let import_index = module.resolved_import_count;
            let import_directive = module.imports.get_elt(import_index);
            alt self.resolve_import_for_module(module, import_directive) {
                Failed {
                    // We presumably emitted an error. Continue.
                    self.session.span_err(import_directive.span,
                                          ~"failed to resolve import");
                }
                Indeterminate {
                    // Bail out. We'll come around next time.
                    break;
                }
                Success(()) {
                    // Good. Continue.
                }
            }

            module.resolved_import_count += 1u;
        }
    }

    /**
     * Attempts to resolve the given import. The return value indicates
     * failure if we're certain the name does not exist, indeterminate if we
     * don't know whether the name exists at the moment due to other
     * currently-unresolved imports, or success if we know the name exists.
     * If successful, the resolved bindings are written into the module.
     */
    fn resolve_import_for_module(module: @Module,
                                 import_directive: @ImportDirective)
                              -> ResolveResult<()> {

        let mut resolution_result;
        let module_path = import_directive.module_path;

        #debug("(resolving import for module) resolving import `%s::...` in \
                `%s`",
               *(*self.atom_table).atoms_to_str((*module_path).get()),
               self.module_to_str(module));

        // One-level renaming imports of the form `import foo = bar;` are
        // handled specially.

        if (*module_path).len() == 0u {
            resolution_result =
                self.resolve_one_level_renaming_import(module,
                                                       import_directive);
        } else {
            // First, resolve the module path for the directive, if necessary.
            alt self.resolve_module_path_for_import(module,
                                                    module_path,
                                                    NoXray,
                                                    import_directive.span) {

                Failed {
                    resolution_result = Failed;
                }
                Indeterminate {
                    resolution_result = Indeterminate;
                }
                Success(containing_module) {
                    // We found the module that the target is contained
                    // within. Attempt to resolve the import within it.

                    alt *import_directive.subclass {
                        SingleImport(target, source) {
                            resolution_result =
                                self.resolve_single_import(module,
                                                           containing_module,
                                                           target,
                                                           source);
                        }
                        GlobImport {
                            let span = import_directive.span;
                            resolution_result =
                                self.resolve_glob_import(module,
                                                         containing_module,
                                                         span);
                        }
                    }
                }
            }
        }

        // Decrement the count of unresolved imports.
        alt resolution_result {
            Success(()) {
                assert self.unresolved_imports >= 1u;
                self.unresolved_imports -= 1u;
            }
            _ {
                // Nothing to do here; just return the error.
            }
        }

        // Decrement the count of unresolved globs if necessary. But only if
        // the resolution result is indeterminate -- otherwise we'll stop
        // processing imports here. (See the loop in
        // resolve_imports_for_module.)

        if resolution_result != Indeterminate {
            alt *import_directive.subclass {
                GlobImport {
                    assert module.glob_count >= 1u;
                    module.glob_count -= 1u;
                }
                SingleImport(*) {
                    // Ignore.
                }
            }
        }

        ret resolution_result;
    }

    fn resolve_single_import(module: @Module, containing_module: @Module,
                             target: Atom, source: Atom)
                          -> ResolveResult<()> {

        #debug("(resolving single import) resolving `%s` = `%s::%s` from \
                `%s`",
               *(*self.atom_table).atom_to_str(target),
               self.module_to_str(containing_module),
               *(*self.atom_table).atom_to_str(source),
               self.module_to_str(module));

        if !self.name_is_exported(containing_module, source) {
            #debug("(resolving single import) name `%s` is unexported",
                   *(*self.atom_table).atom_to_str(source));
            ret Failed;
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
        alt containing_module.children.find(source) {
            none {
                // Continue.
            }
            some(child_name_bindings) {
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

        alt (module_result, value_result, type_result, impl_result) {
            (BoundResult(*), BoundResult(*), BoundResult(*),
             BoundImplResult(*)) {
                // Continue.
            }
            _ {
                // If there is an unresolved glob at this point in the
                // containing module, bail out. We don't know enough to be
                // able to resolve this import.

                if containing_module.glob_count > 0u {
                    #debug("(resolving single import) unresolved glob; \
                            bailing out");
                    ret Indeterminate;
                }

                // Now search the exported imports within the containing
                // module.

                alt containing_module.import_resolutions.find(source) {
                    none {
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
                                == 0u {

                        fn get_binding(import_resolution: @ImportResolution,
                                       namespace: Namespace)
                                    -> NamespaceResult {

                            alt (*import_resolution).
                                    target_for_namespace(namespace) {
                                none {
                                    ret UnboundResult;
                                }
                                some(target) {
                                    import_resolution.used = true;
                                    ret BoundResult(target.target_module,
                                                    target.bindings);
                                }
                            }
                        }

                        fn get_import_binding(import_resolution:
                                              @ImportResolution)
                                           -> ImplNamespaceResult {

                            if (*import_resolution.impl_target).len() == 0u {
                                ret UnboundImplResult;
                            }
                            ret BoundImplResult(import_resolution.
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
                    some(_) {
                        // The import is unresolved. Bail out.
                        #debug("(resolving single import) unresolved import; \
                                bailing out");
                        ret Indeterminate;
                    }
                }
            }
        }

        // We've successfully resolved the import. Write the results in.
        assert module.import_resolutions.contains_key(target);
        let import_resolution = module.import_resolutions.get(target);

        alt module_result {
            BoundResult(target_module, name_bindings) {
                #debug("(resolving single import) found module binding");
                import_resolution.module_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult {
                #debug("(resolving single import) didn't find module \
                        binding");
            }
            UnknownResult {
                fail ~"module result should be known at this point";
            }
        }
        alt value_result {
            BoundResult(target_module, name_bindings) {
                import_resolution.value_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult { /* Continue. */ }
            UnknownResult {
                fail ~"value result should be known at this point";
            }
        }
        alt type_result {
            BoundResult(target_module, name_bindings) {
                import_resolution.type_target =
                    some(Target(target_module, name_bindings));
            }
            UnboundResult { /* Continue. */ }
            UnknownResult {
                fail ~"type result should be known at this point";
            }
        }
        alt impl_result {
            BoundImplResult(targets) {
                for (*targets).each |target| {
                    (*import_resolution.impl_target).push(target);
                }
            }
            UnboundImplResult { /* Continue. */ }
            UnknownImplResult {
                fail ~"impl result should be known at this point";
            }
        }

        assert import_resolution.outstanding_references >= 1u;
        import_resolution.outstanding_references -= 1u;

        #debug("(resolving single import) successfully resolved import");
        ret Success(());
    }

    /**
     * Resolves a glob import. Note that this function cannot fail; it either
     * succeeds or bails out (as importing * from an empty module or a module
     * that exports nothing is valid).
     */
    fn resolve_glob_import(module: @Module,
                           containing_module: @Module,
                           span: span)
                        -> ResolveResult<()> {

        // This function works in a highly imperative manner; it eagerly adds
        // everything it can to the list of import resolutions of the module
        // node.

        // We must bail out if the node has unresolved imports of any kind
        // (including globs).

        if !(*containing_module).all_imports_resolved() {
            #debug("(resolving glob import) target module has unresolved \
                    imports; bailing out");
            ret Indeterminate;
        }

        assert containing_module.glob_count == 0u;

        // Add all resolved imports from the containing module.
        for containing_module.import_resolutions.each
                |atom, target_import_resolution| {

            if !self.name_is_exported(containing_module, atom) {
                #debug("(resolving glob import) name `%s` is unexported",
                       *(*self.atom_table).atom_to_str(atom));
                again;
            }

            #debug("(resolving glob import) writing module resolution \
                    %? into `%s`",
                   is_none(target_import_resolution.module_target),
                   self.module_to_str(module));

            // Here we merge two import resolutions.
            alt module.import_resolutions.find(atom) {
                none {
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

                    module.import_resolutions.insert
                        (atom, new_import_resolution);
                }
                some(dest_import_resolution) {
                    // Merge the two import resolutions at a finer-grained
                    // level.

                    alt copy target_import_resolution.module_target {
                        none {
                            // Continue.
                        }
                        some(module_target) {
                            dest_import_resolution.module_target =
                                some(copy module_target);
                        }
                    }
                    alt copy target_import_resolution.value_target {
                        none {
                            // Continue.
                        }
                        some(value_target) {
                            dest_import_resolution.value_target =
                                some(copy value_target);
                        }
                    }
                    alt copy target_import_resolution.type_target {
                        none {
                            // Continue.
                        }
                        some(type_target) {
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
                #debug("(resolving glob import) name `%s` is unexported",
                       *(*self.atom_table).atom_to_str(atom));
                again;
            }

            let mut dest_import_resolution;
            alt module.import_resolutions.find(atom) {
                none {
                    // Create a new import resolution from this child.
                    dest_import_resolution = @ImportResolution(span);
                    module.import_resolutions.insert
                        (atom, dest_import_resolution);
                }
                some(existing_import_resolution) {
                    dest_import_resolution = existing_import_resolution;
                }
            }


            #debug("(resolving glob import) writing resolution `%s` in `%s` \
                    to `%s`",
                   *(*self.atom_table).atom_to_str(atom),
                   self.module_to_str(containing_module),
                   self.module_to_str(module));

            // Merge the child item into the import resolution.
            if (*name_bindings).defined_in_namespace(ModuleNS) {
                #debug("(resolving glob import) ... for module target");
                dest_import_resolution.module_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(ValueNS) {
                #debug("(resolving glob import) ... for value target");
                dest_import_resolution.value_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(TypeNS) {
                #debug("(resolving glob import) ... for type target");
                dest_import_resolution.type_target =
                    some(Target(containing_module, name_bindings));
            }
            if (*name_bindings).defined_in_namespace(ImplNS) {
                #debug("(resolving glob import) ... for impl target");
                (*dest_import_resolution.impl_target).push
                    (@Target(containing_module, name_bindings));
            }
        }

        #debug("(resolving glob import) successfully resolved import");
        ret Success(());
    }

    fn resolve_module_path_from_root(module: @Module,
                                     module_path: @dvec<Atom>,
                                     index: uint,
                                     xray: XrayFlag,
                                     span: span)
                                  -> ResolveResult<@Module> {

        let mut search_module = module;
        let mut index = index;
        let module_path_len = (*module_path).len();

        // Resolve the module part of the path. This does not involve looking
        // upward though scope chains; we simply resolve names directly in
        // modules as we go.

        while index < module_path_len {
            let name = (*module_path).get_elt(index);
            alt self.resolve_name_in_module(search_module, name, ModuleNS,
                                            xray) {

                Failed {
                    self.session.span_err(span, ~"unresolved name");
                    ret Failed;
                }
                Indeterminate {
                    #debug("(resolving module path for import) module \
                            resolution is indeterminate: %s",
                            *(*self.atom_table).atom_to_str(name));
                    ret Indeterminate;
                }
                Success(target) {
                    alt target.bindings.module_def {
                        NoModuleDef {
                            // Not a module.
                            self.session.span_err(span,
                                                  #fmt("not a module: %s",
                                                       *(*self.atom_table).
                                                         atom_to_str(name)));
                            ret Failed;
                        }
                        ModuleDef(module) {
                            search_module = module;
                        }
                    }
                }
            }

            index += 1u;
        }

        ret Success(search_module);
    }

    /**
     * Attempts to resolve the module part of an import directive rooted at
     * the given module.
     */
    fn resolve_module_path_for_import(module: @Module,
                                      module_path: @dvec<Atom>,
                                      xray: XrayFlag,
                                      span: span)
                                   -> ResolveResult<@Module> {

        let module_path_len = (*module_path).len();
        assert module_path_len > 0u;

        #debug("(resolving module path for import) processing `%s` rooted at \
               `%s`",
               *(*self.atom_table).atoms_to_str((*module_path).get()),
               self.module_to_str(module));

        // The first element of the module path must be in the current scope
        // chain.

        let first_element = (*module_path).get_elt(0u);
        let mut search_module;
        alt self.resolve_module_in_lexical_scope(module, first_element) {
            Failed {
                self.session.span_err(span, ~"unresolved name");
                ret Failed;
            }
            Indeterminate {
                #debug("(resolving module path for import) indeterminate; \
                        bailing");
                ret Indeterminate;
            }
            Success(resulting_module) {
                search_module = resulting_module;
            }
        }

        ret self.resolve_module_path_from_root(search_module,
                                               module_path,
                                               1u,
                                               xray,
                                               span);
    }

    fn resolve_item_in_lexical_scope(module: @Module,
                                     name: Atom,
                                     namespace: Namespace)
                                  -> ResolveResult<Target> {

        #debug("(resolving item in lexical scope) resolving `%s` in \
                namespace %? in `%s`",
               *(*self.atom_table).atom_to_str(name),
               namespace,
               self.module_to_str(module));

        // The current module node is handled specially. First, check for
        // its immediate children.

        alt module.children.find(name) {
            some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) {

                ret Success(Target(module, name_bindings));
            }
            some(_) | none { /* Not found; continue. */ }
        }

        // Now check for its import directives. We don't have to have resolved
        // all its imports in the usual way; this is because chains of
        // adjacent import statements are processed as though they mutated the
        // current scope.

        alt module.import_resolutions.find(name) {
            none {
                // Not found; continue.
            }
            some(import_resolution) {
                alt (*import_resolution).target_for_namespace(namespace) {
                    none {
                        // Not found; continue.
                        #debug("(resolving item in lexical scope) found \
                                import resolution, but not in namespace %?",
                               namespace);
                    }
                    some(target) {
                        import_resolution.used = true;
                        ret Success(copy target);
                    }
                }
            }
        }

        // Finally, proceed up the scope chain looking for parent modules.
        let mut search_module = module;
        loop {
            // Go to the next parent.
            alt search_module.parent_link {
                NoParentLink {
                    // No more parents. This module was unresolved.
                    #debug("(resolving item in lexical scope) unresolved \
                            module");
                    ret Failed;
                }
                ModuleParentLink(parent_module_node, _) |
                BlockParentLink(parent_module_node, _) {
                    search_module = parent_module_node;
                }
            }

            // Resolve the name in the parent module.
            alt self.resolve_name_in_module(search_module, name, namespace,
                                            Xray) {
                Failed {
                    // Continue up the search chain.
                }
                Indeterminate {
                    // We couldn't see through the higher scope because of an
                    // unresolved import higher up. Bail.

                    #debug("(resolving item in lexical scope) indeterminate \
                            higher scope; bailing");
                    ret Indeterminate;
                }
                Success(target) {
                    // We found the module.
                    ret Success(copy target);
                }
            }
        }
    }

    fn resolve_module_in_lexical_scope(module: @Module, name: Atom)
                                    -> ResolveResult<@Module> {

        alt self.resolve_item_in_lexical_scope(module, name, ModuleNS) {
            Success(target) {
                alt target.bindings.module_def {
                    NoModuleDef {
                        #error("!!! (resolving module in lexical scope) module
                                wasn't actually a module!");
                        ret Failed;
                    }
                    ModuleDef(module) {
                        ret Success(module);
                    }
                }
            }
            Indeterminate {
                #debug("(resolving module in lexical scope) indeterminate; \
                        bailing");
                ret Indeterminate;
            }
            Failed {
                #debug("(resolving module in lexical scope) failed to \
                        resolve");
                ret Failed;
            }
        }
    }

    fn name_is_exported(module: @Module, name: Atom) -> bool {
        ret module.exported_names.size() == 0u ||
                module.exported_names.contains_key(name);
    }

    /**
     * Attempts to resolve the supplied name in the given module for the
     * given namespace. If successful, returns the target corresponding to
     * the name.
     */
    fn resolve_name_in_module(module: @Module,
                              name: Atom,
                              namespace: Namespace,
                              xray: XrayFlag)
                           -> ResolveResult<Target> {

        #debug("(resolving name in module) resolving `%s` in `%s`",
               *(*self.atom_table).atom_to_str(name),
               self.module_to_str(module));

        if xray == NoXray && !self.name_is_exported(module, name) {
            #debug("(resolving name in module) name `%s` is unexported",
                   *(*self.atom_table).atom_to_str(name));
            ret Failed;
        }

        // First, check the direct children of the module.
        alt module.children.find(name) {
            some(name_bindings)
                    if (*name_bindings).defined_in_namespace(namespace) {

                #debug("(resolving name in module) found node as child");
                ret Success(Target(module, name_bindings));
            }
            some(_) | none {
                // Continue.
            }
        }

        // Next, check the module's imports. If the module has a glob, then
        // we bail out; we don't know its imports yet.

        if module.glob_count > 0u {
            #debug("(resolving name in module) module has glob; bailing out");
            ret Indeterminate;
        }

        // Otherwise, we check the list of resolved imports.
        alt module.import_resolutions.find(name) {
            some(import_resolution) {
                if import_resolution.outstanding_references != 0u {
                    #debug("(resolving name in module) import unresolved; \
                            bailing out");
                    ret Indeterminate;
                }

                alt (*import_resolution).target_for_namespace(namespace) {
                    none {
                        #debug("(resolving name in module) name found, but \
                                not in namespace %?",
                               namespace);
                    }
                    some(target) {
                        #debug("(resolving name in module) resolved to \
                                import");
                        import_resolution.used = true;
                        ret Success(copy target);
                    }
                }
            }
            none {
                // Continue.
            }
        }

        // We're out of luck.
        #debug("(resolving name in module) failed to resolve %s",
               *(*self.atom_table).atom_to_str(name));
        ret Failed;
    }

    /**
     * Resolves a one-level renaming import of the kind `import foo = bar;`
     * This needs special handling, as, unlike all of the other imports, it
     * needs to look in the scope chain for modules and non-modules alike.
     */
    fn resolve_one_level_renaming_import(module: @Module,
                                         import_directive: @ImportDirective)
                                      -> ResolveResult<()> {

        let mut target_name;
        let mut source_name;
        alt *import_directive.subclass {
            SingleImport(target, source) {
                target_name = target;
                source_name = source;
            }
            GlobImport {
                fail ~"found `import *`, which is invalid";
            }
        }

        #debug("(resolving one-level naming result) resolving import `%s` = \
                `%s` in `%s`",
                *(*self.atom_table).atom_to_str(target_name),
                *(*self.atom_table).atom_to_str(source_name),
                self.module_to_str(module));

        // Find the matching items in the lexical scope chain for every
        // namespace. If any of them come back indeterminate, this entire
        // import is indeterminate.

        let mut module_result;
        #debug("(resolving one-level naming result) searching for module");
        alt self.resolve_item_in_lexical_scope(module,
                                               source_name,
                                               ModuleNS) {

            Failed {
                #debug("(resolving one-level renaming import) didn't find \
                        module result");
                module_result = none;
            }
            Indeterminate {
                #debug("(resolving one-level renaming import) module result \
                        is indeterminate; bailing");
                ret Indeterminate;
            }
            Success(name_bindings) {
                #debug("(resolving one-level renaming import) module result \
                        found");
                module_result = some(copy name_bindings);
            }
        }

        let mut value_result;
        #debug("(resolving one-level naming result) searching for value");
        alt self.resolve_item_in_lexical_scope(module,
                                               source_name,
                                               ValueNS) {

            Failed {
                #debug("(resolving one-level renaming import) didn't find \
                        value result");
                value_result = none;
            }
            Indeterminate {
                #debug("(resolving one-level renaming import) value result \
                        is indeterminate; bailing");
                ret Indeterminate;
            }
            Success(name_bindings) {
                #debug("(resolving one-level renaming import) value result \
                        found");
                value_result = some(copy name_bindings);
            }
        }

        let mut type_result;
        #debug("(resolving one-level naming result) searching for type");
        alt self.resolve_item_in_lexical_scope(module,
                                               source_name,
                                               TypeNS) {

            Failed {
                #debug("(resolving one-level renaming import) didn't find \
                        type result");
                type_result = none;
            }
            Indeterminate {
                #debug("(resolving one-level renaming import) type result is \
                        indeterminate; bailing");
                ret Indeterminate;
            }
            Success(name_bindings) {
                #debug("(resolving one-level renaming import) type result \
                        found");
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
        #debug("(resolving one-level naming result) searching for impl");
        alt self.resolve_item_in_lexical_scope(module,
                                               source_name,
                                               ImplNS) {

            Failed {
                #debug("(resolving one-level renaming import) didn't find \
                        impl result");
                impl_result = none;
            }
            Indeterminate {
                #debug("(resolving one-level renaming import) impl result is \
                        indeterminate; bailing");
                ret Indeterminate;
            }
            Success(name_bindings) {
                #debug("(resolving one-level renaming import) impl result \
                        found");
                impl_result = some(@copy name_bindings);
            }
        }

        // If nothing at all was found, that's an error.
        if is_none(module_result) && is_none(value_result) &&
                is_none(type_result) && is_none(impl_result) {

            self.session.span_err(import_directive.span,
                                  ~"unresolved import");
            ret Failed;
        }

        // Otherwise, proceed and write in the bindings.
        alt module.import_resolutions.find(target_name) {
            none {
                fail ~"(resolving one-level renaming import) reduced graph \
                      construction or glob importing should have created the \
                      import resolution name by now";
            }
            some(import_resolution) {
                #debug("(resolving one-level renaming import) writing module \
                        result %? for `%s` into `%s`",
                       is_none(module_result),
                       *(*self.atom_table).atom_to_str(target_name),
                       self.module_to_str(module));

                import_resolution.module_target = module_result;
                import_resolution.value_target = value_result;
                import_resolution.type_target = type_result;

                alt impl_result {
                    none {
                        // Nothing to do.
                    }
                    some(impl_result) {
                        (*import_resolution.impl_target).push(impl_result);
                    }
                }

                assert import_resolution.outstanding_references >= 1u;
                import_resolution.outstanding_references -= 1u;
            }
        }

        #debug("(resolving one-level renaming import) successfully resolved");
        ret Success(());
    }

    fn report_unresolved_imports(module: @Module) {
        let index = module.resolved_import_count;
        let import_count = module.imports.len();
        if index != import_count {
            self.session.span_err(module.imports.get_elt(index).span,
                                  ~"unresolved import");
        }

        // Descend into children and anonymous children.
        for module.children.each |_name, child_node| {
            alt (*child_node).get_module_if_available() {
                none {
                    // Continue.
                }
                some(child_module) {
                    self.report_unresolved_imports(child_module);
                }
            }
        }

        for module.anonymous_children.each |_name, module| {
            self.report_unresolved_imports(module);
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

    fn record_exports_for_module_subtree(module: @Module) {
        // If this isn't a local crate, then bail out. We don't need to record
        // exports for local crates.

        alt module.def_id {
            some(def_id) if def_id.crate == local_crate {
                // OK. Continue.
            }
            none {
                // Record exports for the root module.
            }
            some(_) {
                // Bail out.
                #debug("(recording exports for module subtree) not recording \
                        exports for `%s`",
                       self.module_to_str(module));
                ret;
            }
        }

        self.record_exports_for_module(module);

        for module.children.each |_atom, child_name_bindings| {
            alt (*child_name_bindings).get_module_if_available() {
                none {
                    // Nothing to do.
                }
                some(child_module) {
                    self.record_exports_for_module_subtree(child_module);
                }
            }
        }

        for module.anonymous_children.each |_node_id, child_module| {
            self.record_exports_for_module_subtree(child_module);
        }
    }

    fn record_exports_for_module(module: @Module) {
        for module.exported_names.each |name, node_id| {
            let mut exports = ~[];
            for self.namespaces.each |namespace| {
                // Ignore impl namespaces; they cause the original resolve
                // to fail.

                if namespace == ImplNS {
                    again;
                }

                alt self.resolve_definition_of_name_in_module(module,
                                                              name,
                                                              namespace,
                                                              Xray) {
                    NoNameDefinition {
                        // Nothing to do.
                    }
                    ChildNameDefinition(target_def) {
                        vec::push(exports, {
                            reexp: false,
                            id: def_id_of_def(target_def)
                        });
                    }
                    ImportNameDefinition(target_def) {
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

    fn build_impl_scopes_for_module_subtree(module: @Module) {
        // If this isn't a local crate, then bail out. We don't need to
        // resolve implementations for external crates.

        alt module.def_id {
            some(def_id) if def_id.crate == local_crate {
                // OK. Continue.
            }
            none {
                // Resolve implementation scopes for the root module.
            }
            some(_) {
                // Bail out.
                #debug("(building impl scopes for module subtree) not \
                        resolving implementations for `%s`",
                       self.module_to_str(module));
                ret;
            }
        }

        self.build_impl_scope_for_module(module);

        for module.children.each |_atom, child_name_bindings| {
            alt (*child_name_bindings).get_module_if_available() {
                none {
                    // Nothing to do.
                }
                some(child_module) {
                    self.build_impl_scopes_for_module_subtree(child_module);
                }
            }
        }

        for module.anonymous_children.each |_node_id, child_module| {
            self.build_impl_scopes_for_module_subtree(child_module);
        }
    }

    fn build_impl_scope_for_module(module: @Module) {
        let mut impl_scope = ~[];

        #debug("(building impl scope for module) processing module %s (%?)",
               self.module_to_str(module),
               copy module.def_id);

        // Gather up all direct children implementations in the module.
        for module.children.each |_impl_name, child_name_bindings| {
            if child_name_bindings.impl_defs.len() >= 1u {
                impl_scope += child_name_bindings.impl_defs;
            }
        }

        #debug("(building impl scope for module) found %u impl(s) as direct \
                children",
               impl_scope.len());

        // Gather up all imports.
        for module.import_resolutions.each |_impl_name, import_resolution| {
            for (*import_resolution.impl_target).each |impl_target| {
                #debug("(building impl scope for module) found impl def");
                impl_scope += impl_target.bindings.impl_defs;
            }
        }

        #debug("(building impl scope for module) found %u impl(s) in total",
               impl_scope.len());

        // Determine the parent's implementation scope.
        let mut parent_impl_scopes;
        alt module.parent_link {
            NoParentLink {
                parent_impl_scopes = @nil;
            }
            ModuleParentLink(parent_module_node, _) |
            BlockParentLink(parent_module_node, _) {
                parent_impl_scopes = parent_module_node.impl_scopes;
            }
        }

        // Create the new implementation scope, if it was nonempty, and chain
        // it up to the parent.

        if impl_scope.len() >= 1u {
            module.impl_scopes = @cons(@impl_scope, parent_impl_scopes);
        } else {
            module.impl_scopes = parent_impl_scopes;
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
        alt name {
            none {
                // Nothing to do.
            }
            some(name) {
                alt orig_module.children.find(name) {
                    none {
                        #debug("!!! (with scope) didn't find `%s` in `%s`",
                               *(*self.atom_table).atom_to_str(name),
                               self.module_to_str(orig_module));
                    }
                    some(name_bindings) {
                        alt (*name_bindings).get_module_if_available() {
                            none {
                                #debug("!!! (with scope) didn't find module \
                                        for `%s` in `%s`",
                                       *(*self.atom_table).atom_to_str(name),
                                       self.module_to_str(orig_module));
                            }
                            some(module) {
                                self.current_module = module;
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

        alt def_like {
            dl_def(d @ def_local(*)) | dl_def(d @ def_upvar(*)) |
            dl_def(d @ def_arg(*)) | dl_def(d @ def_binding(*)) {
                def = d;
                is_ty_param = false;
            }
            dl_def(d @ def_ty_param(*)) {
                def = d;
                is_ty_param = true;
            }
            dl_def(d @ def_self(*))
                    if allow_capturing_self == DontAllowCapturingSelf {
                def = d;
                is_ty_param = false;
            }
            _ {
                ret some(def_like);
            }
        }

        let mut rib_index = rib_index + 1u;
        while rib_index < (*ribs).len() {
            let rib = (*ribs).get_elt(rib_index);
            alt rib.kind {
                NormalRibKind {
                    // Nothing to do. Continue.
                }
                FunctionRibKind(function_id) {
                    if !is_ty_param {
                        def = def_upvar(def_id_of_def(def).node,
                                        @def,
                                        function_id);
                    }
                }
                OpaqueFunctionRibKind {
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

                    ret none;
                }
            }

            rib_index += 1u;
        }

        ret some(dl_def(def));
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
            alt rib.bindings.find(name) {
                some(def_like) {
                    ret self.upvarify(ribs, i, def_like, span,
                                      allow_capturing_self);
                }
                none {
                    // Continue.
                }
            }
        }

        ret none;
    }

    // XXX: This shouldn't be unsafe!
    fn resolve_crate() unsafe {
        #debug("(resolving crate) starting");

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
        #debug("(resolving item) resolving %s", *item.ident);

        // Items with the !resolve_unexported attribute are X-ray contexts.
        // This is used to allow the test runner to run unexported tests.
        let orig_xray_flag = self.xray_context;
        if contains_name(attr_metas(item.attrs), ~"!resolve_unexported") {
            self.xray_context = Xray;
        }

        alt item.node {
            item_enum(_, type_parameters) |
            item_ty(_, type_parameters) {
                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, item.id, 0u,
                                           NormalRibKind))
                        || {

                    visit_item(item, (), visitor);
                }
            }

            item_impl(type_parameters, implemented_traits, self_type,
                      methods) {

                self.resolve_implementation(item.id, item.span,
                                            type_parameters,
                                            implemented_traits,
                                            self_type, methods, visitor);
            }

            item_trait(type_parameters, methods) {
                // Create a new rib for the self type.
                let self_type_rib = @Rib(NormalRibKind);
                (*self.type_ribs).push(self_type_rib);
                self_type_rib.bindings.insert(self.self_atom,
                                              dl_def(def_self(item.id)));

                // Create a new rib for the interface-wide type parameters.
                do self.with_type_parameter_rib
                        (HasTypeParameters(&type_parameters, item.id, 0u,
                                           NormalRibKind)) {

                    self.resolve_type_parameters(type_parameters, visitor);

                    for methods.each |method| {
                        // Create a new rib for the method-specific type
                        // parameters.
                        //
                        // XXX: Do we need a node ID here?

                        alt method {
                          required(ty_m) {
                            do self.with_type_parameter_rib
                                (HasTypeParameters(&ty_m.tps,
                                                   item.id,
                                                   type_parameters.len(),
                                                   NormalRibKind)) {

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
                          provided(m) {
                              self.resolve_method(NormalRibKind,
                                                  m,
                                                  type_parameters.len(),
                                                  visitor)
                          }
                        }
                    }
                }

                (*self.type_ribs).pop();
            }

            item_class(ty_params, interfaces, class_members, constructor,
                       optional_destructor) {

                self.resolve_class(item.id,
                                   @copy ty_params,
                                   interfaces,
                                   class_members,
                                   constructor,
                                   optional_destructor,
                                   visitor);
            }

            item_mod(module) {
                let atom = (*self.atom_table).intern(item.ident);
                do self.with_scope(some(atom)) {
                    self.resolve_module(module, item.span, item.ident,
                                        item.id, visitor);
                }
            }

            item_foreign_mod(foreign_module) {
                let atom = (*self.atom_table).intern(item.ident);
                do self.with_scope(some(atom)) {
                    for foreign_module.items.each |foreign_item| {
                        alt foreign_item.node {
                            foreign_item_fn(_, type_parameters) {
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

            item_fn(fn_decl, ty_params, block) {
                // If this is the main function, we must record it in the
                // session.
                //
                // For speed, we put the string comparison last in this chain
                // of conditionals.

                if !self.session.building_library &&
                        is_none(self.session.main_fn) &&
                        str::eq(*item.ident, ~"main") {

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

            item_const(*) {
                visit_item(item, (), visitor);
            }

          item_mac(*) {
            fail ~"item macros unimplemented"
          }
        }

        self.xray_context = orig_xray_flag;
    }

    fn with_type_parameter_rib(type_parameters: TypeParameters, f: fn()) {
        alt type_parameters {
            HasTypeParameters(type_parameters, node_id, initial_index,
                              rib_kind) {

                let function_type_rib = @Rib(rib_kind);
                (*self.type_ribs).push(function_type_rib);

                for (*type_parameters).eachi |index, type_parameter| {
                    let name =
                        (*self.atom_table).intern(type_parameter.ident);
                    let def_like = dl_def(def_ty_param
                        (local_def(type_parameter.id),
                         index + initial_index));
                    (*function_type_rib).bindings.insert(name, def_like);
                }
            }

            NoTypeParameters {
                // Nothing to do.
            }
        }

        f();

        alt type_parameters {
            HasTypeParameters(type_parameters, _, _, _) {
                (*self.type_ribs).pop();
            }

            NoTypeParameters {
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
        alt capture_clause {
            NoCaptureClause {
                // Nothing to do.
            }
            HasCaptureClause(capture_clause) {
                // Resolve each captured item.
                for (*capture_clause).each |capture_item| {
                    alt self.resolve_identifier(capture_item.name,
                                                ValueNS,
                                                true,
                                                capture_item.span) {
                        none {
                            self.session.span_err(capture_item.span,
                                                  ~"unresolved name in \
                                                   capture clause");
                        }
                        some(def) {
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
            alt type_parameters {
                NoTypeParameters {
                    // Continue.
                }
                HasTypeParameters(type_parameters, _, _, _) {
                    self.resolve_type_parameters(*type_parameters, visitor);
                }
            }

            // Add self to the rib, if necessary.
            alt self_binding {
                NoSelfBinding {
                    // Nothing to do.
                }
                HasSelfBinding(self_node_id) {
                    let def_like = dl_def(def_self(self_node_id));
                    (*function_value_rib).bindings.insert(self.self_atom,
                                                          def_like);
                }
            }

            // Add each argument to the rib.
            alt optional_declaration {
                none {
                    // Nothing to do.
                }
                some(declaration) {
                    for declaration.inputs.each |argument| {
                        let name = (*self.atom_table).intern(argument.ident);
                        let def_like = dl_def(def_arg(argument.id,
                                                      argument.mode));
                        (*function_value_rib).bindings.insert(name, def_like);

                        self.resolve_type(argument.ty, visitor);

                        #debug("(resolving function) recorded argument `%s`",
                               *(*self.atom_table).atom_to_str(name));
                    }

                    self.resolve_type(declaration.output, visitor);
                }
            }

            // Resolve the function body.
            self.resolve_block(block, visitor);

            #debug("(resolving function) leaving function");
        }

        (*self.value_ribs).pop();
    }

    fn resolve_type_parameters(type_parameters: ~[ty_param],
                               visitor: ResolveVisitor) {

        for type_parameters.each |type_parameter| {
            for (*type_parameter.bounds).each |bound| {
                alt bound {
                    bound_copy | bound_send | bound_const | bound_owned {
                        // Nothing to do.
                    }
                    bound_trait(interface_type) {
                        self.resolve_type(interface_type, visitor);
                    }
                }
            }
        }
    }

    fn resolve_class(id: node_id,
                     type_parameters: @~[ty_param],
                     interfaces: ~[@trait_ref],
                     class_members: ~[@class_member],
                     constructor: class_ctor,
                     optional_destructor: option<class_dtor>,
                     visitor: ResolveVisitor) {

        // Add a type into the def map. This is needed to prevent an ICE in
        // ty::impl_trait.

        // If applicable, create a rib for the type parameters.
        let outer_type_parameter_count = (*type_parameters).len();
        let borrowed_type_parameters: &~[ty_param] = &*type_parameters;
        do self.with_type_parameter_rib(HasTypeParameters
                                        (borrowed_type_parameters, id, 0u,
                                         NormalRibKind))
                || {

            // Resolve the type parameters.
            self.resolve_type_parameters(*type_parameters, visitor);

            // Resolve implemented interfaces.
            for interfaces.each |interface| {
                alt self.resolve_path(interface.path, TypeNS, true, visitor) {
                    none {
                        self.session.span_err(interface.path.span,
                                              ~"attempt to implement a \
                                               nonexistent interface");
                    }
                    some(def) {
                        // Write a mapping from the interface ID to the
                        // definition of the interface into the definition
                        // map.

                        #debug("(resolving class) found trait def: %?", def);

                        self.record_def(interface.ref_id, def);

                        // XXX: This is wrong but is needed for tests to
                        // pass.

                        self.record_def(id, def);
                    }
                }
            }

            // Resolve methods.
            for class_members.each |class_member| {
                alt class_member.node {
                    class_method(method) {
                      self.resolve_method(NormalRibKind,
                                          method,
                                          outer_type_parameter_count,
                                          visitor);
                    }
                    instance_var(_, field_type, _, _, _) {
                        self.resolve_type(field_type, visitor);
                    }
                }
            }

            // Resolve the constructor.
            self.resolve_function(NormalRibKind,
                                  some(@constructor.node.dec),
                                  NoTypeParameters,
                                  constructor.node.body,
                                  HasSelfBinding(constructor.node.self_id),
                                  NoCaptureClause,
                                  visitor);


            // Resolve the destructor, if applicable.
            alt optional_destructor {
                none {
                    // Nothing to do.
                }
                some(destructor) {
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
        self.resolve_function(rib_kind,
                              some(@method.decl),
                              type_parameters,
                              method.body,
                              HasSelfBinding(method.self_id),
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

            // Resolve the interface reference, if necessary.
            let original_trait_refs = self.current_trait_refs;
            if trait_references.len() >= 1 {
                let mut new_trait_refs = @dvec();
                for trait_references.each |trait_reference| {
                    alt self.resolve_path(trait_reference.path, TypeNS, true,
                                          visitor) {
                        none {
                            self.session.span_err(span,
                                                  ~"attempt to implement an \
                                                    unknown trait");
                        }
                        some(def) {
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

                let borrowed_type_parameters = &method.tps;
                self.resolve_function(NormalRibKind,
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
            }

            // Restore the original trait references.
            self.current_trait_refs = original_trait_refs;
        }
    }

    fn resolve_module(module: _mod, span: span, _name: ident, id: node_id,
                      visitor: ResolveVisitor) {

        // Write the implementations in scope into the module metadata.
        #debug("(resolving module) resolving module ID %d", id);
        self.impl_map.insert(id, self.current_module.impl_scopes);

        visit_mod(module, span, id, (), visitor);
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
        alt local.node.init {
            none {
                // Nothing to do.
            }
            some(initializer) {
                self.resolve_expr(initializer.expr, visitor);
            }
        }

        // Resolve the pattern.
        self.resolve_pattern(local.node.pat, IrrefutableMode, mutability,
                             none, visitor);
    }

    fn num_bindings(pat: @pat) -> uint {
      pat_util::pat_binding_ids(self.def_map, pat).len()
    }

    fn warn_var_patterns(arm: arm) {
      /*
        The idea here is that an arm like:
           alpha | beta
        where alpha is a variant and beta is an identifier that
        might refer to a variant that's not in scope will result
        in a confusing error message. Showing that beta actually binds a
        new variable might help.
       */
      for arm.pats.each |p| {
         do pat_util::pat_bindings(self.def_map, p) |_id, sp, pth| {
             self.session.span_note(sp, #fmt("Treating %s as a variable \
               binding, because it does not denote any variant in scope",
                                             path_to_str(pth)));
         }
      };
    }
    fn check_consistent_bindings(arm: arm) {
      if arm.pats.len() == 0 { ret; }
      let good = self.num_bindings(arm.pats[0]);
      for arm.pats.each() |p: @pat| {
        if self.num_bindings(p) != good {
          self.session.span_err(p.span,
             ~"inconsistent number of bindings");
          self.warn_var_patterns(arm);
          break;
        };
      };
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
        #debug("(resolving block) entering block");
        (*self.value_ribs).push(@Rib(NormalRibKind));

        // Move down in the graph, if there's an anonymous module rooted here.
        let orig_module = self.current_module;
        alt self.current_module.anonymous_children.find(block.node.id) {
            none { /* Nothing to do. */ }
            some(anonymous_module) {
                #debug("(resolving block) found anonymous module, moving \
                        down");
                self.current_module = anonymous_module;
            }
        }

        // Descend into the block.
        visit_block(block, (), visitor);

        // Move back up.
        self.current_module = orig_module;

        (*self.value_ribs).pop();
        #debug("(resolving block) leaving block");
    }

    fn resolve_type(ty: @ty, visitor: ResolveVisitor) {
        alt ty.node {
            // Like path expressions, the interpretation of path types depends
            // on whether the path has multiple elements in it or not.

            ty_path(path, path_id) {
                // This is a path in the type namespace. Walk through scopes
                // scopes looking for it.

                let mut result_def;
                alt self.resolve_path(path, TypeNS, true, visitor) {
                    some(def) {
                        #debug("(resolving type) resolved `%s` to type",
                               *path.idents.last());
                        result_def = some(def);
                    }
                    none {
                        result_def = none;
                    }
                }

                alt result_def {
                    some(_) {
                        // Continue.
                    }
                    none {
                        // Check to see whether the name is a primitive type.
                        if path.idents.len() == 1u {
                            let name =
                                (*self.atom_table).intern(path.idents.last());

                            alt self.primitive_type_table
                                    .primitive_types
                                    .find(name) {

                                some(primitive_type) {
                                    result_def =
                                        some(def_prim_ty(primitive_type));
                                }
                                none {
                                    // Continue.
                                }
                            }
                        }
                    }
                }

                alt copy result_def {
                    some(def) {
                        // Write the result into the def map.
                        #debug("(resolving type) writing resolution for `%s` \
                                (id %d)",
                               connect(path.idents.map(|x| *x), ~"::"),
                               path_id);
                        self.record_def(path_id, def);
                    }
                    none {
                        self.session.span_err
                            (ty.span, #fmt("use of undeclared type name `%s`",
                                           connect(path.idents.map(|x| *x),
                                                   ~"::")));
                    }
                }
            }

            _ {
                // Just resolve embedded types.
                visit_ty(ty, (), visitor);
            }
        }
    }

    fn resolve_pattern(pattern: @pat,
                       mode: PatternBindingMode,
                       mutability: Mutability,
                       bindings_list: option<hashmap<Atom,()>>,
                       visitor: ResolveVisitor) {

        do walk_pat(pattern) |pattern| {
            alt pattern.node {
                pat_ident(path, _)
                        if !path.global && path.idents.len() == 1u {

                    // The meaning of pat_ident with no type parameters
                    // depends on whether an enum variant with that name is in
                    // scope. The probing lookup has to be careful not to emit
                    // spurious errors. Only matching patterns (alt) can match
                    // nullary variants. For binding patterns (let), matching
                    // such a variant is simply disallowed (since it's rarely
                    // what you want).

                    let atom = (*self.atom_table).intern(path.idents[0]);

                    alt self.resolve_enum_variant_or_const(atom) {
                        FoundEnumVariant(def) if mode == RefutableMode {
                            #debug("(resolving pattern) resolving `%s` to \
                                    enum variant",
                                   *path.idents[0]);

                            self.record_def(pattern.id, def);
                        }
                        FoundEnumVariant(_) {
                            self.session.span_err(pattern.span,
                                                  #fmt("declaration of `%s` \
                                                        shadows an enum \
                                                        that's in scope",
                                                       *(*self.atom_table).
                                                            atom_to_str
                                                            (atom)));
                        }
                        FoundConst {
                            self.session.span_err(pattern.span,
                                                  ~"pattern variable \
                                                   conflicts with a constant \
                                                   in scope");
                        }
                        EnumVariantOrConstNotFound {
                            #debug("(resolving pattern) binding `%s`",
                                   *path.idents[0]);

                            let is_mutable = mutability == Mutable;

                            let mut def;
                            alt mode {
                                RefutableMode {
                                    // For pattern arms, we must use
                                    // `def_binding` definitions.

                                    def = def_binding(pattern.id);
                                }
                                IrrefutableMode {
                                    // But for locals, we use `def_local`.
                                    def = def_local(pattern.id, is_mutable);
                                }
                            }

                            // Record the definition so that later passes
                            // will be able to distinguish variants from
                            // locals in patterns.

                            self.record_def(pattern.id, def);

                            // Add the binding to the local ribs, if it
                            // doesn't already exist in the bindings list. (We
                            // must not add it if it's in the bindings list
                            // because that breaks the assumptions later
                            // passes make about or-patterns.)

                            alt bindings_list {
                                some(bindings_list)
                                        if !bindings_list.contains_key(atom) {
                                    let last_rib = (*self.value_ribs).last();
                                    last_rib.bindings.insert(atom,
                                                             dl_def(def));
                                    bindings_list.insert(atom, ());
                                }
                                some(_) {
                                    // Do nothing.
                                }
                                none {
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

                pat_ident(path, _) | pat_enum(path, _) {
                    // These two must be enum variants.
                    alt self.resolve_path(path, ValueNS, false, visitor) {
                        some(def @ def_variant(*)) {
                            self.record_def(pattern.id, def);
                        }
                        some(_) {
                            self.session.span_err(path.span,
                                                  #fmt("not an enum \
                                                        variant: %s",
                                                       *path.idents.last()));
                        }
                        none {
                            self.session.span_err(path.span,
                                                  ~"unresolved enum variant");
                        }
                    }

                    // Check the types in the path pattern.
                    for path.types.each |ty| {
                        self.resolve_type(ty, visitor);
                    }
                }

                pat_lit(expr) {
                    self.resolve_expr(expr, visitor);
                }

                pat_range(first_expr, last_expr) {
                    self.resolve_expr(first_expr, visitor);
                    self.resolve_expr(last_expr, visitor);
                }

                _ {
                    // Nothing to do.
                }
            }
        }
    }

    fn resolve_enum_variant_or_const(name: Atom)
                                  -> EnumVariantOrConstResolution {

        alt self.resolve_item_in_lexical_scope(self.current_module,
                                               name,
                                               ValueNS) {

            Success(target) {
                alt target.bindings.value_def {
                    none {
                        fail ~"resolved name in the value namespace to a set \
                              of name bindings with no def?!";
                    }
                    some(def @ def_variant(*)) {
                        ret FoundEnumVariant(def);
                    }
                    some(def_const(*)) {
                        ret FoundConst;
                    }
                    some(_) {
                        ret EnumVariantOrConstNotFound;
                    }
                }
            }

            Indeterminate {
                fail ~"unexpected indeterminate result";
            }

            Failed {
                ret EnumVariantOrConstNotFound;
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
            ret self.resolve_crate_relative_path(path,
                                                 self.xray_context,
                                                 namespace);
        }

        if path.idents.len() > 1u {
            ret self.resolve_module_relative_path(path,
                                                  self.xray_context,
                                                  namespace);
        }

        ret self.resolve_identifier(path.idents.last(),
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
            alt self.resolve_identifier_in_local_ribs(identifier,
                                                      namespace,
                                                      span) {
                some(def) {
                    ret some(def);
                }
                none {
                    // Continue.
                }
            }
        }

        ret self.resolve_item_by_identifier_in_lexical_scope(identifier,
                                                             namespace);
    }

    // XXX: Merge me with resolve_name_in_module?
    fn resolve_definition_of_name_in_module(containing_module: @Module,
                                            name: Atom,
                                            namespace: Namespace,
                                            xray: XrayFlag)
                                         -> NameDefinition {

        if xray == NoXray && !self.name_is_exported(containing_module, name) {
            #debug("(resolving definition of name in module) name `%s` is \
                    unexported",
                   *(*self.atom_table).atom_to_str(name));
            ret NoNameDefinition;
        }

        // First, search children.
        alt containing_module.children.find(name) {
            some(child_name_bindings) {
                alt (*child_name_bindings).def_for_namespace(namespace) {
                    some(def) {
                        // Found it. Stop the search here.
                        ret ChildNameDefinition(def);
                    }
                    none {
                        // Continue.
                    }
                }
            }
            none {
                // Continue.
            }
        }

        // Next, search import resolutions.
        alt containing_module.import_resolutions.find(name) {
            some(import_resolution) {
                alt (*import_resolution).target_for_namespace(namespace) {
                    some(target) {
                        alt (*target.bindings).def_for_namespace(namespace) {
                            some(def) {
                                // Found it.
                                import_resolution.used = true;
                                ret ImportNameDefinition(def);
                            }
                            none {
                                // This can happen with external impls, due to
                                // the imperfect way we read the metadata.

                                ret NoNameDefinition;
                            }
                        }
                    }
                    none {
                        ret NoNameDefinition;
                    }
                }
            }
            none {
                ret NoNameDefinition;
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

        ret module_path_atoms;
    }

    fn resolve_module_relative_path(path: @path,
                                    +xray: XrayFlag,
                                    namespace: Namespace)
                                 -> option<def> {

        let module_path_atoms = self.intern_module_part_of_path(path);

        let mut containing_module;
        alt self.resolve_module_path_for_import(self.current_module,
                                                module_path_atoms,
                                                xray,
                                                path.span) {

            Failed {
                self.session.span_err(path.span,
                                      #fmt("use of undeclared module `%s`",
                                            *(*self.atom_table).atoms_to_str
                                              ((*module_path_atoms).get())));
                ret none;
            }

            Indeterminate {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) {
                containing_module = resulting_module;
            }
        }

        let name = (*self.atom_table).intern(path.idents.last());
        alt self.resolve_definition_of_name_in_module(containing_module,
                                                      name,
                                                      namespace,
                                                      xray) {
            NoNameDefinition {
                // We failed to resolve the name. Report an error.
                self.session.span_err(path.span,
                                      #fmt("unresolved name: %s::%s",
                                           *(*self.atom_table).atoms_to_str
                                               ((*module_path_atoms).get()),
                                           *(*self.atom_table).atom_to_str
                                               (name)));
                ret none;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) {
                ret some(def);
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
        alt self.resolve_module_path_from_root(root_module,
                                               module_path_atoms,
                                               0u,
                                               xray,
                                               path.span) {

            Failed {
                self.session.span_err(path.span,
                                      #fmt("use of undeclared module `::%s`",
                                            *(*self.atom_table).atoms_to_str
                                              ((*module_path_atoms).get())));
                ret none;
            }

            Indeterminate {
                fail ~"indeterminate unexpected";
            }

            Success(resulting_module) {
                containing_module = resulting_module;
            }
        }

        let name = (*self.atom_table).intern(path.idents.last());
        alt self.resolve_definition_of_name_in_module(containing_module,
                                                      name,
                                                      namespace,
                                                      xray) {
            NoNameDefinition {
                // We failed to resolve the name. Report an error.
                self.session.span_err(path.span,
                                      #fmt("unresolved name: %s::%s",
                                           *(*self.atom_table).atoms_to_str
                                               ((*module_path_atoms).get()),
                                           *(*self.atom_table).atom_to_str
                                               (name)));
                ret none;
            }
            ChildNameDefinition(def) | ImportNameDefinition(def) {
                ret some(def);
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
        alt namespace {
            ValueNS {
                search_result = self.search_ribs(self.value_ribs, name, span,
                                                 DontAllowCapturingSelf);
            }
            TypeNS {
                search_result = self.search_ribs(self.type_ribs, name, span,
                                                 AllowCapturingSelf);
            }
            ModuleNS | ImplNS {
                fail ~"module or impl namespaces do not have local ribs";
            }
        }

        alt copy search_result {
            some(dl_def(def)) {
                #debug("(resolving path in local ribs) resolved `%s` to \
                        local: %?",
                       *(*self.atom_table).atom_to_str(name),
                       def);
                ret some(def);
            }
            some(dl_field) | some(dl_impl(_)) | none {
                ret none;
            }
        }
    }

    fn resolve_item_by_identifier_in_lexical_scope(ident: ident,
                                                   namespace: Namespace)
                                                -> option<def> {

        let name = (*self.atom_table).intern(ident);

        // Check the items.
        alt self.resolve_item_in_lexical_scope(self.current_module,
                                               name,
                                               namespace) {

            Success(target) {
                alt (*target.bindings).def_for_namespace(namespace) {
                    none {
                        fail ~"resolved name in a namespace to a set of name \
                              bindings with no def for that namespace?!";
                    }
                    some(def) {
                        #debug("(resolving item path in lexical scope) \
                                resolved `%s` to item",
                               *(*self.atom_table).atom_to_str(name));
                        ret some(def);
                    }
                }
            }
            Indeterminate {
                fail ~"unexpected indeterminate result";
            }
            Failed {
                ret none;
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
        alt expr.node {
            // The interpretation of paths depends on whether the path has
            // multiple elements in it or not.

            expr_path(path) {
                // This is a local path in the value namespace. Walk through
                // scopes looking for it.

                alt self.resolve_path(path, ValueNS, true, visitor) {
                    some(def) {
                        // Write the result into the def map.
                        #debug("(resolving expr) resolved `%s`",
                               connect(path.idents.map(|x| *x), ~"::"));
                        self.record_def(expr.id, def);
                    }
                    none {
                        self.session.span_err(expr.span,
                                              #fmt("unresolved name: %s",
                                              connect(path.idents.map(|x| *x),
                                                      ~"::")));
                    }
                }

                visit_expr(expr, (), visitor);
            }

            expr_fn(_, fn_decl, block, capture_clause) |
            expr_fn_block(fn_decl, block, capture_clause) {
                self.resolve_function(FunctionRibKind(expr.id),
                                      some(@fn_decl),
                                      NoTypeParameters,
                                      block,
                                      NoSelfBinding,
                                      HasCaptureClause(capture_clause),
                                      visitor);
            }

            _ {
                visit_expr(expr, (), visitor);
            }
        }
    }

    fn record_impls_for_expr_if_necessary(expr: @expr) {
        alt expr.node {
            expr_field(*) | expr_path(*) | expr_cast(*) | expr_binary(*) |
            expr_unary(*) | expr_assign_op(*) | expr_index(*) {
                self.impl_map.insert(expr.id,
                                     self.current_module.impl_scopes);
            }
            expr_new(container, _, _) {
                self.impl_map.insert(container.id,
                                     self.current_module.impl_scopes);
            }
            _ {
                // Nothing to do.
            }
        }
    }

    fn record_candidate_traits_for_expr_if_necessary(expr: @expr) {
        alt expr.node {
            expr_field(_, ident, _) {
                let atom = (*self.atom_table).intern(ident);
                let traits = self.search_for_traits_containing_method(atom);
                self.trait_map.insert(expr.id, traits);
            }
            _ {
                // Nothing to do.
                //
                // XXX: Handle more here... operator overloading, placement
                // new, etc.
            }
        }
    }

    fn search_for_traits_containing_method(name: Atom) -> @dvec<def_id> {
        let found_traits = @dvec();
        let mut search_module = self.current_module;
        loop {
            // Look for the current trait.
            alt copy self.current_trait_refs {
                some(trait_def_ids) {
                    for trait_def_ids.each |trait_def_id| {
                        self.add_trait_info_if_containing_method
                            (found_traits, trait_def_id, name);
                    }
                }
                none {
                    // Nothing to do.
                }
            }

            // Look for trait children.
            for search_module.children.each |_name, child_name_bindings| {
                alt child_name_bindings.def_for_namespace(TypeNS) {
                    some(def_ty(trait_def_id)) {
                        self.add_trait_info_if_containing_method(found_traits,
                                                                 trait_def_id,
                                                                 name);
                    }
                    some(_) | none {
                        // Continue.
                    }
                }
            }

            // Look for imports.
            for search_module.import_resolutions.each
                    |_atom, import_resolution| {

                alt import_resolution.target_for_namespace(TypeNS) {
                    none {
                        // Continue.
                    }
                    some(target) {
                        alt target.bindings.def_for_namespace(TypeNS) {
                            some(def_ty(trait_def_id)) {
                                self.add_trait_info_if_containing_method
                                    (found_traits, trait_def_id, name);
                            }
                            some(_) | none {
                                // Continue.
                            }
                        }
                    }
                }
            }

            // Move to the next parent.
            alt search_module.parent_link {
                NoParentLink {
                    // Done.
                    break;
                }
                ModuleParentLink(parent_module, _) |
                BlockParentLink(parent_module, _) {
                    search_module = parent_module;
                }
            }
        }

        ret found_traits;
    }

    fn add_trait_info_if_containing_method(found_traits: @dvec<def_id>,
                                           trait_def_id: def_id,
                                           name: Atom) {

        alt self.trait_info.find(trait_def_id) {
            some(trait_info) if trait_info.contains_key(name) {
                #debug("(adding trait info if containing method) found trait \
                        %d:%d for method '%s'",
                       trait_def_id.crate,
                       trait_def_id.node,
                       *(*self.atom_table).atom_to_str(name));
                (*found_traits).push(trait_def_id);
            }
            some(_) | none {
                // Continue.
            }
        }
    }

    fn record_def(node_id: node_id, def: def) {
        #debug("(recording def) recording %? for %?", def, node_id);
        self.def_map.insert(node_id, def);
    }

    //
    // Unused import checking
    //
    // Although this is a lint pass, it lives in here because it depends on
    // resolve data structures.
    //

    fn check_for_unused_imports_if_necessary() {
        if self.unused_import_lint_level == ignore {
            ret;
        }

        let root_module = (*self.graph_root).get_module();
        self.check_for_unused_imports_in_module_subtree(root_module);
    }

    fn check_for_unused_imports_in_module_subtree(module: @Module) {
        // If this isn't a local crate, then bail out. We don't need to check
        // for unused imports in external crates.

        alt module.def_id {
            some(def_id) if def_id.crate == local_crate {
                // OK. Continue.
            }
            none {
                // Check for unused imports in the root module.
            }
            some(_) {
                // Bail out.
                #debug("(checking for unused imports in module subtree) not \
                        checking for unused imports for `%s`",
                       self.module_to_str(module));
                ret;
            }
        }

        self.check_for_unused_imports_in_module(module);

        for module.children.each |_atom, child_name_bindings| {
            alt (*child_name_bindings).get_module_if_available() {
                none {
                    // Nothing to do.
                }
                some(child_module) {
                    self.check_for_unused_imports_in_module_subtree
                        (child_module);
                }
            }
        }

        for module.anonymous_children.each |_node_id, child_module| {
            self.check_for_unused_imports_in_module_subtree(child_module);
        }
    }

    fn check_for_unused_imports_in_module(module: @Module) {
        for module.import_resolutions.each |_impl_name, import_resolution| {
            if !import_resolution.used {
                alt self.unused_import_lint_level {
                    warn {
                        self.session.span_warn(import_resolution.span,
                                               ~"unused import");
                    }
                    error {
                        self.session.span_err(import_resolution.span,
                                              ~"unused import");
                    }
                    ignore {
                        self.session.span_bug(import_resolution.span,
                                              ~"shouldn't be here if lint \
                                               pass is ignored");
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
    fn module_to_str(module: @Module) -> ~str {
        let atoms = dvec();
        let mut current_module = module;
        loop {
            alt current_module.parent_link {
                NoParentLink {
                    break;
                }
                ModuleParentLink(module, name) {
                    atoms.push(name);
                    current_module = module;
                }
                BlockParentLink(module, node_id) {
                    atoms.push((*self.atom_table).intern(@~"<opaque>"));
                    current_module = module;
                }
            }
        }

        if atoms.len() == 0u {
            ret ~"???";
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

        ret string;
    }

    fn dump_module(module: @Module) {
        #debug("Dump of module `%s`:", self.module_to_str(module));

        #debug("Children:");
        for module.children.each |name, _child| {
            #debug("* %s", *(*self.atom_table).atom_to_str(name));
        }

        #debug("Import resolutions:");
        for module.import_resolutions.each |name, import_resolution| {
            let mut module_repr;
            alt (*import_resolution).target_for_namespace(ModuleNS) {
                none { module_repr = ~""; }
                some(target) {
                    module_repr = ~" module:?";
                    // XXX
                }
            }

            let mut value_repr;
            alt (*import_resolution).target_for_namespace(ValueNS) {
                none { value_repr = ~""; }
                some(target) {
                    value_repr = ~" value:?";
                    // XXX
                }
            }

            let mut type_repr;
            alt (*import_resolution).target_for_namespace(TypeNS) {
                none { type_repr = ~""; }
                some(target) {
                    type_repr = ~" type:?";
                    // XXX
                }
            }

            let mut impl_repr;
            alt (*import_resolution).target_for_namespace(ImplNS) {
                none { impl_repr = ~""; }
                some(target) {
                    impl_repr = ~" impl:?";
                    // XXX
                }
            }

            #debug("* %s:%s%s%s%s",
                   *(*self.atom_table).atom_to_str(name),
                   module_repr, value_repr, type_repr, impl_repr);
        }
    }

    fn dump_impl_scopes(impl_scopes: ImplScopes) {
        #debug("Dump of impl scopes:");

        let mut i = 0u;
        let mut impl_scopes = impl_scopes;
        loop {
            alt *impl_scopes {
                cons(impl_scope, rest_impl_scopes) {
                    #debug("Impl scope %u:", i);

                    for (*impl_scope).each |implementation| {
                        #debug("Impl: %s", *implementation.ident);
                    }

                    i += 1u;
                    impl_scopes = rest_impl_scopes;
                }
                nil {
                    break;
                }
            }
        }
    }
}

/// Entry point to crate resolution.
fn resolve_crate(session: session, ast_map: ASTMap, crate: @crate)
              -> { def_map: DefMap,
                   exp_map: ExportMap,
                   impl_map: ImplMap,
                   trait_map: TraitMap } {

    let resolver = @Resolver(session, ast_map, crate);
    (*resolver).resolve(resolver);
    ret {
        def_map: resolver.def_map,
        exp_map: resolver.export_map,
        impl_map: resolver.impl_map,
        trait_map: resolver.trait_map
    };
}

