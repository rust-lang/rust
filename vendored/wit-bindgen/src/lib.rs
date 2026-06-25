//! > **⚠️ Warning ⚠️**: this crate is an internal-only crate for the Wasmtime
//! > project and is not intended for general use. APIs are not strictly
//! > reviewed for safety and usage outside of Wasmtime may have bugs. If
//! > you're interested in using this feel free to file an issue on the
//! > Wasmtime repository to start a discussion about doing so, but otherwise
//! > be aware that your usage of this crate is not supported.

#![allow(warnings)]

use crate::rust::{RustGenerator, TypeMode, to_rust_ident, to_rust_upper_camel_case};
use crate::types::{TypeInfo, Types};
use anyhow::bail;
use heck::*;
use indexmap::{IndexMap, IndexSet};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{Read, Write};
use std::mem;
use std::process::{Command, Stdio};
use wit_parser::*;

macro_rules! uwrite {
    ($dst:expr, $($arg:tt)*) => {
        write!($dst, $($arg)*).unwrap()
    };
}

macro_rules! uwriteln {
    ($dst:expr, $($arg:tt)*) => {
        writeln!($dst, $($arg)*).unwrap()
    };
}

mod config;
mod rust;
mod source;
mod types;

pub use config::{FunctionConfig, FunctionFilter, FunctionFlags};
use source::Source;

#[derive(Clone)]
enum InterfaceName {
    /// This interface was remapped using `with` to some other Rust code.
    Remapped {
        /// This is the `::`-separated string which is the path to the mapped
        /// item relative to the root of the `bindgen!` macro invocation.
        ///
        /// This path currently starts with `__with_name$N` and will then
        /// optionally have `::` projections through to the actual item
        /// depending on how `with` was configured.
        name_at_root: String,

        /// This is currently only used for exports and is the relative path to
        /// where this mapped name would be located if `with` were not
        /// specified. Basically it's the same as the `Path` variant of this
        /// enum if the mapping weren't present.
        local_path: Vec<String>,
    },

    /// This interface is generated in the module hierarchy specified.
    ///
    /// The path listed here is the path, from the root of the `bindgen!` macro,
    /// to where this interface is generated.
    Path(Vec<String>),
}

#[derive(Default)]
struct Wasmtime {
    src: Source,
    opts: Opts,
    /// A list of all interfaces which were imported by this world.
    import_interfaces: IndexMap<InterfaceId, ImportInterface>,
    world_import_functions: Vec<Function>,
    world_implements_interfaces: Vec<(String, InterfaceId)>,
    interfaces_for_implements: HashMap<InterfaceId, InterfaceId>,
    exports: Exports,
    types: Types,
    sizes: SizeAlign,
    interface_names: HashMap<InterfaceId, InterfaceName>,
    trappable_errors: IndexMap<TypeId, String>,
    // Track the with options that were used. Remapped interfaces provided via `with`
    // are required to be used.
    used_with_opts: HashSet<String>,
    // Modules generated for `named_imports` options.
    named_import_modules: Vec<(String, InterfaceName)>,
    world_link_options: LinkOptionsBuilder,
    interface_link_options: HashMap<InterfaceId, LinkOptionsBuilder>,
}

struct ImportInterface {
    contents: String,
    name: InterfaceName,
    all_func_flags: FunctionFlags,
    in_world: bool,
}

#[derive(Default)]
struct Exports {
    fields: BTreeMap<String, ExportField>,
    modules: Vec<(String, InterfaceName)>,
    funcs: Vec<String>,
}

struct ExportField {
    ty: String,
    ty_index: String,
    load: String,
    get_index: String,
}

#[derive(Default, Debug, Clone, Copy)]
pub enum Ownership {
    /// Generated types will be composed entirely of owning fields, regardless
    /// of whether they are used as parameters to guest exports or not.
    #[default]
    Owning,

    /// Generated types used as parameters to guest exports will be "deeply
    /// borrowing", i.e. contain references rather than owned values when
    /// applicable.
    Borrowing {
        /// Whether or not to generate "duplicate" type definitions for a single
        /// WIT type if necessary, for example if it's used as both an import
        /// and an export, or if it's used both as a parameter to an export and
        /// a return value from an export.
        duplicate_if_necessary: bool,
    },
}

#[derive(Default, Debug, Clone)]
pub struct Opts {
    /// Whether or not `rustfmt` is executed to format generated code.
    pub rustfmt: bool,

    /// A list of "trappable errors" which are used to replace the `E` in
    /// `result<T, E>` found in WIT.
    pub trappable_error_type: Vec<TrappableError>,

    /// Whether to generate owning or borrowing type definitions.
    pub ownership: Ownership,

    /// Whether or not to generate code for only the interfaces of this wit file or not.
    pub only_interfaces: bool,

    /// Remapping of interface names to rust module names.
    /// TODO: is there a better type to use for the value of this map?
    pub with: HashMap<String, String>,

    /// Interfaces for which to generate an additional set of "named imports"
    /// bindings.
    pub named_imports: HashMap<String, String>,

    /// Additional derive attributes to add to generated types. If using in a CLI, this flag can be
    /// specified multiple times to add multiple attributes.
    ///
    /// These derive attributes will be added to any generated structs or enums
    pub additional_derive_attributes: Vec<String>,

    /// Evaluate to a string literal containing the generated code rather than the generated tokens
    /// themselves. Mostly useful for Wasmtime internal debugging and development.
    pub stringify: bool,

    /// Temporary option to skip `impl<T: Trait> Trait for &mut T` for the
    /// `wasmtime-wasi` crate while that's given a chance to update its b
    /// indings.
    pub skip_mut_forwarding_impls: bool,

    /// Indicates that the `T` in `Store<T>` should be send even if async is not
    /// enabled.
    ///
    /// This is helpful when sync bindings depend on generated functions from
    /// async bindings as is the case with WASI in-tree.
    pub require_store_data_send: bool,

    /// Path to the `wasmtime` crate if it's not the default path.
    pub wasmtime_crate: Option<String>,

    /// Whether to use `anyhow::Result` for trappable host-defined function
    /// imports.
    ///
    /// By default, `wasmtime::Result` is used instead of `anyhow::Result`.
    ///
    /// When enabled, the generated code requires the `"anyhow"` cargo feature
    /// to also be enabled in the `wasmtime` crate.
    pub anyhow: bool,

    /// If true, write the generated bindings to a file for better error
    /// messages from `rustc`.
    ///
    /// This can also be toggled via the `WASMTIME_DEBUG_BINDGEN` environment
    /// variable, but that will affect _all_ `bindgen!` macro invocations (and
    /// can sometimes lead to one invocation overwriting another in unpredictable
    /// ways), whereas this option lets you specify it on a case-by-case basis.
    pub debug: bool,

    /// TODO
    pub imports: FunctionConfig,
    /// TODO
    pub exports: FunctionConfig,
}

#[derive(Debug, Clone)]
pub struct TrappableError {
    /// Full path to the error, such as `wasi:io/streams.error`.
    pub wit_path: String,

    /// The name, in Rust, of the error type to generate.
    pub rust_type_name: String,
}

impl Opts {
    pub fn generate(&self, resolve: &mut Resolve, world: WorldId) -> anyhow::Result<String> {
        resolve.generate_nominal_type_ids(world);
        // TODO: Should we refine this test to inspect only types reachable from
        // the specified world?
        if !cfg!(feature = "component-model-async")
            && resolve
                .types
                .iter()
                .any(|(_, ty)| matches!(ty.kind, TypeDefKind::Future(_) | TypeDefKind::Stream(_)))
        {
            anyhow::bail!(
                "must enable `component-model-async` feature when using WIT files \
                 containing future, stream, or error-context types"
            );
        }

        let mut r = Wasmtime::default();
        r.sizes.fill(resolve);
        r.opts = self.clone();
        r.populate_world_and_interface_options(resolve, world);
        r.generate(resolve, world)
    }
}

#[derive(Copy, Clone, PartialEq)]
enum InterfaceKind {
    Import,
    Export,
    Named,
}

impl Wasmtime {
    fn populate_world_and_interface_options(&mut self, resolve: &Resolve, world: WorldId) {
        self.world_link_options.add_world(resolve, &world);

        for (_, import) in resolve.worlds[world].imports.iter() {
            match import {
                WorldItem::Interface { id, .. } => {
                    let mut o = LinkOptionsBuilder::default();
                    o.add_interface(resolve, id);
                    self.interface_link_options.insert(*id, o);
                }
                WorldItem::Function(_) | WorldItem::Type { .. } => {}
            }
        }
    }

    fn generate_interface_name(
        &mut self,
        resolve: &Resolve,
        id: InterfaceId,
        name: &WorldKey,
        interface_kind: InterfaceKind,
    ) -> Vec<String> {
        let mut path = Vec::new();
        match interface_kind {
            InterfaceKind::Import => {}
            InterfaceKind::Export => path.push("exports".to_string()),
            InterfaceKind::Named => path.push("named_imports".to_string()),
        }
        match name {
            WorldKey::Name(name) => {
                path.push(name.to_snake_case());
            }
            WorldKey::Interface(_) => {
                let iface = &resolve.interfaces[id];
                let pkgname = &resolve.packages[iface.package.unwrap()].name;
                path.push(pkgname.namespace.to_snake_case());
                path.push(self.name_package_module(resolve, iface.package.unwrap()));
                path.push(to_rust_ident(iface.name.as_ref().unwrap()));
            }
        }
        path
    }

    fn name_interface(
        &mut self,
        resolve: &Resolve,
        id: InterfaceId,
        name: &WorldKey,
        interface_kind: InterfaceKind,
    ) -> bool {
        let local_path = self.generate_interface_name(resolve, id, name, interface_kind);
        let entry = if !matches!(interface_kind, InterfaceKind::Named)
            && let Some(name_at_root) = self.lookup_replacement(resolve, name, None)
        {
            InterfaceName::Remapped {
                name_at_root,
                local_path,
            }
        } else {
            InterfaceName::Path(local_path)
        };

        let remapped = matches!(entry, InterfaceName::Remapped { .. });
        let prev = self.interface_names.insert(id, entry);
        assert!(prev.is_none());
        remapped
    }

    /// If the package `id` is the only package with its namespace/name combo
    /// then pass through the name unmodified. If, however, there are multiple
    /// versions of this package then the package module is going to get version
    /// information.
    fn name_package_module(&self, resolve: &Resolve, id: PackageId) -> String {
        let pkg = &resolve.packages[id];
        let versions_with_same_name = resolve
            .packages
            .iter()
            .filter_map(|(_, p)| {
                if p.name.namespace == pkg.name.namespace && p.name.name == pkg.name.name {
                    Some(&p.name.version)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let base = pkg.name.name.to_snake_case();
        if versions_with_same_name.len() == 1 {
            return base;
        }

        let version = match &pkg.name.version {
            Some(version) => version,
            // If this package didn't have a version then don't mangle its name
            // and other packages with the same name but with versions present
            // will have their names mangled.
            None => return base,
        };

        // Here there's multiple packages with the same name that differ only in
        // version, so the version needs to be mangled into the Rust module name
        // that we're generating. This in theory could look at all of
        // `versions_with_same_name` and produce a minimal diff, e.g. for 0.1.0
        // and 0.2.0 this could generate "foo1" and "foo2", but for now
        // a simpler path is chosen to generate "foo0_1_0" and "foo0_2_0".
        let version = version
            .to_string()
            .replace('.', "_")
            .replace('-', "_")
            .replace('+', "_")
            .to_snake_case();
        format!("{base}{version}")
    }

    fn generate(&mut self, resolve: &Resolve, id: WorldId) -> anyhow::Result<String> {
        self.types.analyze(resolve, id);

        self.world_link_options.write_struct(&mut self.src);

        // Resolve the `trappable_error_type` configuration values to `TypeId`
        // values. This is done by iterating over each `trappable_error_type`
        // and then locating the interface that it corresponds to as well as the
        // type within that interface.
        //
        // Note that `LookupItem::InterfaceNoPop` is used here as the full
        // hierarchical behavior of `lookup_keys` isn't used as the interface
        // must be named here.
        'outer: for (i, te) in self.opts.trappable_error_type.iter().enumerate() {
            let error_name = format!("_TrappableError{i}");
            for (id, iface) in resolve.interfaces.iter() {
                for (key, projection) in lookup_keys(
                    resolve,
                    &WorldKey::Interface(id),
                    LookupItem::InterfaceNoPop,
                ) {
                    assert!(projection.is_empty());

                    // If `wit_path` looks like `{key}.{type_name}` where
                    // `type_name` is a type within `iface` then we've found a
                    // match. Otherwise continue to the next lookup key if there
                    // is one, and failing that continue to the next interface.
                    let suffix = match te.wit_path.strip_prefix(&key) {
                        Some(s) => s,
                        None => continue,
                    };
                    let suffix = match suffix.strip_prefix('.') {
                        Some(s) => s,
                        None => continue,
                    };
                    if let Some(id) = iface.types.get(suffix) {
                        uwriteln!(self.src, "type {error_name} = {};", te.rust_type_name);
                        let prev = self.trappable_errors.insert(*id, error_name);
                        assert!(prev.is_none());
                        continue 'outer;
                    }
                }
            }

            bail!(
                "failed to locate a WIT error type corresponding to the \
                 `trappable_error_type` name `{}` provided",
                te.wit_path
            )
        }

        // Convert all entries in `with` as relative to the root of where the
        // macro itself is invoked. This emits a `pub use` to bring the name
        // into scope under an "anonymous name" which then replaces the `with`
        // map entry.
        let mut with = self.opts.with.iter_mut().collect::<Vec<_>>();
        with.sort();
        for (i, (_k, v)) in with.into_iter().enumerate() {
            let name = format!("__with_name{i}");
            uwriteln!(self.src, "#[doc(hidden)]\npub use {v} as {name};");
            *v = name;
        }

        // Similarly bring the `named_imports` id types into scope under an
        // anonymous name at the root so they can be referenced from the deeply
        // nested `named_imports` module.
        let mut named_imports = self.opts.named_imports.iter_mut().collect::<Vec<_>>();
        named_imports.sort();
        for (i, (_k, v)) in named_imports.into_iter().enumerate() {
            let name = format!("__named_import_name{i}");
            uwriteln!(self.src, "#[doc(hidden)]\npub use {v} as {name};");
            *v = name;
        }

        let world = &resolve.worlds[id];
        for (name, import) in world.imports.iter() {
            if !self.opts.only_interfaces || matches!(import, WorldItem::Interface { .. }) {
                self.import(resolve, name, import);
            }
        }

        for (name, export) in world.exports.iter() {
            if !self.opts.only_interfaces || matches!(export, WorldItem::Interface { .. }) {
                self.export(resolve, name, export);
            }
        }
        self.generate_named_imports(resolve)?;
        self.finish(resolve, id)
    }

    /// Generates the extra "named imports" bindings requested via the
    /// `named_imports` configuration option.
    ///
    /// For each configured interface this generates, into a body stored in
    /// `named_import_modules`, a `Host`/`HostWithStore` trait whose methods take
    /// an extra `&Id` parameter alongside a reflection-based `add_to_linker`.
    /// These are emitted underneath a top-level `named_imports` module by
    /// `finish`.
    fn generate_named_imports(&mut self, resolve: &Resolve) -> anyhow::Result<()> {
        // For all named imports see what interface that lines up with in the
        // `Resolve` which will have bindings generated.
        'outer: for (interface_name, id_type) in self.opts.named_imports.clone() {
            for (id, _iface) in resolve.interfaces.iter() {
                for (key, projection) in lookup_keys(
                    resolve,
                    &WorldKey::Interface(id),
                    LookupItem::InterfaceNoPop,
                ) {
                    assert!(projection.is_empty());
                    if key == interface_name {
                        self.generate_named_import(resolve, id, &interface_name, &id_type)?;
                        continue 'outer;
                    }
                }
            }

            bail!("named imports key {interface_name:?} not found")
        }

        Ok(())
    }

    fn generate_named_import(
        &mut self,
        resolve: &Resolve,
        id: InterfaceId,
        named_import_key: &str,
        id_type: &str,
    ) -> anyhow::Result<()> {
        // Resources are not supported for named imports just yet, it's a bit
        // weird with the resource traits.
        if get_resources(resolve, id).next().is_some() {
            bail!(
                "the interface {named_import_key:?} was specified in \
                `named_imports` but defines a resource, which is not \
                supported"
            );
        }

        let key = WorldKey::Interface(id);
        let mut generator = InterfaceGenerator::new(self, resolve);
        generator.current_interface = Some((id, &key, InterfaceKind::Named));
        let path_to_root = generator.path_to_root();
        generator.named_import_id = Some(format!("{path_to_root}{id_type}"));
        let wt = generator.generator.wasmtime_path();
        generator.src.push_str(&format!(
            "#[allow(unused_imports)] use {wt}::component::__internal::Box;\n"
        ));
        let key_name = resolve.name_world_key(&key);
        generator.generate_add_to_linker(id, &key_name);
        let body = String::from(mem::take(&mut generator.src));
        let interface_name = resolve.interfaces[id].name.as_ref().unwrap();
        let body = format!("pub mod {interface_name} {{\n{body}\n}}");
        let path = self.generate_interface_name(resolve, id, &key, InterfaceKind::Named);
        self.named_import_modules
            .push((body, InterfaceName::Path(path)));
        Ok(())
    }

    fn import(&mut self, resolve: &Resolve, name: &WorldKey, item: &WorldItem) {
        match item {
            WorldItem::Function(func) => {
                self.world_import_functions.push(func.clone());
            }
            WorldItem::Interface { id, .. } => {
                if let WorldKey::Name(kebab) = name
                    && resolve.interfaces[*id].name.is_some()
                {
                    let og_interface = resolve.interfaces[*id].clone_of.unwrap_or(*id);
                    let implements = match self.interfaces_for_implements.get(&og_interface) {
                        Some(id) => *id,
                        None => {
                            self.interfaces_for_implements.insert(og_interface, *id);
                            self.import_interface(resolve, &WorldKey::Interface(*id), *id, false);
                            *id
                        }
                    };
                    self.world_implements_interfaces
                        .push((kebab.to_string(), implements));
                } else {
                    self.import_interface(resolve, name, *id, true);
                }
            }
            WorldItem::Type { id, .. } => {
                let name = match name {
                    WorldKey::Name(name) => name,
                    WorldKey::Interface(_) => unreachable!(),
                };
                let mut generator = InterfaceGenerator::new(self, resolve);
                generator.define_type(name, *id);
                let body = mem::take(&mut generator.src);
                self.src.push_str(&body);
            }
        };
    }

    fn import_interface(
        &mut self,
        resolve: &Resolve,
        name: &WorldKey,
        id: InterfaceId,
        in_world: bool,
    ) {
        let mut generator = InterfaceGenerator::new(self, resolve);

        generator.current_interface = Some((id, name, InterfaceKind::Import));
        let snake = to_rust_ident(&match name {
            WorldKey::Name(s) => s.to_snake_case(),
            WorldKey::Interface(id) => resolve.interfaces[*id]
                .name
                .as_ref()
                .unwrap()
                .to_snake_case(),
        });
        let module = if generator
            .generator
            .name_interface(resolve, id, name, InterfaceKind::Import)
        {
            // If this interface is remapped then that means that it was
            // provided via the `with` key in the bindgen configuration.
            // That means that bindings generation is skipped here. To
            // accommodate future bindgens depending on this bindgen
            // though we still generate a module which reexports the
            // original module. This helps maintain the same output
            // structure regardless of whether `with` is used.
            let name_at_root = match &generator.generator.interface_names[&id] {
                InterfaceName::Remapped { name_at_root, .. } => name_at_root,
                InterfaceName::Path(_) => unreachable!(),
            };
            let path_to_root = generator.path_to_root();
            format!(
                "
                    pub mod {snake} {{
                        #[allow(unused_imports)]
                        pub use {path_to_root}{name_at_root}::*;
                    }}
                "
            )
        } else {
            // If this interface is not remapped then it's time to
            // actually generate bindings here.
            generator.generator.interface_link_options[&id].write_struct(&mut generator.src);
            generator.types(id);
            let key_name = resolve.name_world_key(name);
            generator.generate_add_to_linker(id, &key_name);

            let module = &generator.src[..];
            let wt = generator.generator.wasmtime_path();

            format!(
                "
                    #[allow(clippy::all)]
                    pub mod {snake} {{
                        #[allow(unused_imports)]
                        use {wt}::component::__internal::Box;

                        {module}
                    }}
                "
            )
        };
        let all_func_flags = generator.all_func_flags;
        let prev = self.import_interfaces.insert(
            id,
            ImportInterface {
                contents: module,
                name: self.interface_names[&id].clone(),
                all_func_flags,
                in_world,
            },
        );
        assert!(prev.is_none());

        let interface_path = self.import_interface_path(&id);
        self.interface_link_options[&id].write_impl_from_world(&mut self.src, &interface_path);
    }

    fn export(&mut self, resolve: &Resolve, name: &WorldKey, item: &WorldItem) {
        let wt = self.wasmtime_path();
        let mut generator = InterfaceGenerator::new(self, resolve);
        let field;
        let ty;
        let ty_index;
        let load;
        let get_index;
        match item {
            WorldItem::Function(func) => {
                generator.define_rust_guest_export(resolve, None, func);
                let body = mem::take(&mut generator.src).into();
                load = generator.extract_typed_function(func).1;
                assert!(generator.src.is_empty());
                generator.generator.exports.funcs.push(body);
                ty_index = format!("{wt}::component::ComponentExportIndex");
                field = func_field_name(resolve, func);
                ty = format!("{wt}::component::Func");
                let sig = generator.typedfunc_sig(func, TypeMode::AllBorrowed("'_"));
                let typecheck = format!(
                    "match item {{
                            {wt}::component::types::ComponentItem::ComponentFunc(func) => {{
                                {wt}::error::Context::context(
                                    func.typecheck::<{sig}>(&_instance_type),
                                    \"type-checking export func `{0}`\"
                                )?;
                                index
                            }}
                            _ => Err({wt}::format_err!(\"export `{0}` is not a function\"))?,
                        }}",
                    func.name
                );
                get_index = format!(
                    "{{ let (item, index) = _component.get_export(None, \"{}\")
                        .ok_or_else(|| {wt}::format_err!(\"no export `{0}` found\"))?;
                        {typecheck}
                     }}",
                    func.name
                );
            }
            WorldItem::Type { .. } => unreachable!(),
            WorldItem::Interface { id, .. } => {
                generator
                    .generator
                    .name_interface(resolve, *id, name, InterfaceKind::Export);
                generator.current_interface = Some((*id, name, InterfaceKind::Export));
                generator.types(*id);
                let struct_name = "Guest";
                let iface = &resolve.interfaces[*id];
                let iface_name = match name {
                    WorldKey::Name(name) => name,
                    WorldKey::Interface(_) => iface.name.as_ref().unwrap(),
                };
                uwriteln!(generator.src, "#[derive(Clone)]");
                uwriteln!(generator.src, "pub struct {struct_name} {{");
                for (_, func) in iface.functions.iter() {
                    uwriteln!(
                        generator.src,
                        "{}: {wt}::component::Func,",
                        func_field_name(resolve, func)
                    );
                }
                uwriteln!(generator.src, "}}");

                uwriteln!(generator.src, "#[derive(Clone)]");
                uwriteln!(generator.src, "pub struct {struct_name}Indices {{");
                for (_, func) in iface.functions.iter() {
                    uwriteln!(
                        generator.src,
                        "{}: {wt}::component::ComponentExportIndex,",
                        func_field_name(resolve, func)
                    );
                }
                uwriteln!(generator.src, "}}");

                uwriteln!(generator.src, "impl {struct_name}Indices {{");
                let instance_name = resolve.name_world_key(name);
                uwrite!(
                    generator.src,
                    "
/// Constructor for [`{struct_name}Indices`] which takes a
/// [`Component`]({wt}::component::Component) as input and can be executed
/// before instantiation.
///
/// This constructor can be used to front-load string lookups to find exports
/// within a component.
pub fn new<_T>(
    _instance_pre: &{wt}::component::InstancePre<_T>,
) -> {wt}::Result<{struct_name}Indices> {{
    let instance = _instance_pre.component().get_export_index(None, \"{instance_name}\")
        .ok_or_else(|| {wt}::format_err!(\"no exported instance named `{instance_name}`\"))?;
    let mut lookup = move |name: &str| {{
        _instance_pre.component().get_export_index(Some(&instance), name).ok_or_else(|| {{
            {wt}::format_err!(
                \"instance export `{instance_name}` does \\
                  not have export `{{name}}`\"
            )
        }})
    }};
    let _ = &mut lookup;
                    "
                );
                let mut fields = Vec::new();
                for (_, func) in iface.functions.iter() {
                    let name = func_field_name(resolve, func);
                    uwriteln!(generator.src, "let {name} = lookup(\"{}\")?;", func.name);
                    fields.push(name);
                }
                uwriteln!(generator.src, "Ok({struct_name}Indices {{");
                for name in fields {
                    uwriteln!(generator.src, "{name},");
                }
                uwriteln!(generator.src, "}})");
                uwriteln!(generator.src, "}}"); // end `fn _new`

                uwrite!(
                    generator.src,
                    "
                        pub fn load(
                            &self,
                            mut store: impl {wt}::AsContextMut,
                            instance: &{wt}::component::Instance,
                        ) -> {wt}::Result<{struct_name}> {{
                            let _instance = instance;
                            let _instance_pre = _instance.instance_pre(&store);
                            let _instance_type = _instance_pre.instance_type();
                            let mut store = store.as_context_mut();
                            let _ = &mut store;
                    "
                );
                let mut fields = Vec::new();
                for (_, func) in iface.functions.iter() {
                    let (name, getter) = generator.extract_typed_function(func);
                    uwriteln!(generator.src, "let {name} = {getter};");
                    fields.push(name);
                }
                uwriteln!(generator.src, "Ok({struct_name} {{");
                for name in fields {
                    uwriteln!(generator.src, "{name},");
                }
                uwriteln!(generator.src, "}})");
                uwriteln!(generator.src, "}}"); // end `fn new`
                uwriteln!(generator.src, "}}"); // end `impl {struct_name}Indices`

                uwriteln!(generator.src, "impl {struct_name} {{");
                let mut resource_methods = IndexMap::new();

                for (_, func) in iface.functions.iter() {
                    match func.kind.resource() {
                        None => {
                            generator.define_rust_guest_export(resolve, Some(name), func);
                        }
                        Some(id) => {
                            resource_methods.entry(id).or_insert(Vec::new()).push(func);
                        }
                    }
                }

                for (id, _) in resource_methods.iter() {
                    let name = resolve.types[*id].name.as_ref().unwrap();
                    let snake = name.to_snake_case();
                    let camel = name.to_upper_camel_case();
                    uwriteln!(
                        generator.src,
                        "pub fn {snake}(&self) -> Guest{camel}<'_> {{
                            Guest{camel} {{ funcs: self }}
                        }}"
                    );
                }

                uwriteln!(generator.src, "}}");

                for (id, methods) in resource_methods {
                    let resource_name = resolve.types[id].name.as_ref().unwrap();
                    let camel = resource_name.to_upper_camel_case();
                    uwriteln!(generator.src, "impl Guest{camel}<'_> {{");
                    for method in methods {
                        generator.define_rust_guest_export(resolve, Some(name), method);
                    }
                    uwriteln!(generator.src, "}}");
                }

                let module = &generator.src[..];
                let snake = to_rust_ident(iface_name);

                let module = format!(
                    "
                        #[allow(clippy::all)]
                        pub mod {snake} {{
                            #[allow(unused_imports)]
                            use {wt}::component::__internal::Box;

                            {module}
                        }}
                    "
                );
                let pkgname = match name {
                    WorldKey::Name(_) => None,
                    WorldKey::Interface(_) => {
                        Some(resolve.packages[iface.package.unwrap()].name.clone())
                    }
                };
                self.exports
                    .modules
                    .push((module, self.interface_names[id].clone()));

                let (path, method_name) = match pkgname {
                    Some(pkgname) => (
                        format!(
                            "exports::{}::{}::{snake}::{struct_name}",
                            pkgname.namespace.to_snake_case(),
                            self.name_package_module(resolve, iface.package.unwrap()),
                        ),
                        format!(
                            "{}_{}_{snake}",
                            pkgname.namespace.to_snake_case(),
                            self.name_package_module(resolve, iface.package.unwrap())
                        ),
                    ),
                    None => (format!("exports::{snake}::{struct_name}"), snake.clone()),
                };
                field = format!("interface{}", self.exports.fields.len());
                load = format!("self.{field}.load(&mut store, &_instance)?");
                self.exports.funcs.push(format!(
                    "
                        pub fn {method_name}(&self) -> &{path} {{
                            &self.{field}
                        }}
                    ",
                ));
                ty_index = format!("{path}Indices");
                ty = path;
                get_index = format!("{ty_index}::new(_instance_pre)?");
            }
        }
        let prev = self.exports.fields.insert(
            field,
            ExportField {
                ty,
                ty_index,
                load,
                get_index,
            },
        );
        assert!(prev.is_none());
    }

    fn build_world_struct(&mut self, resolve: &Resolve, world: WorldId) {
        let wt = self.wasmtime_path();
        let world_name = &resolve.worlds[world].name;
        let camel = to_rust_upper_camel_case(&world_name);
        uwriteln!(
            self.src,
            "
/// Auto-generated bindings for a pre-instantiated version of a
/// component which implements the world `{world_name}`.
///
/// This structure is created through [`{camel}Pre::new`] which
/// takes a [`InstancePre`]({wt}::component::InstancePre) that
/// has been created through a [`Linker`]({wt}::component::Linker).
///
/// For more information see [`{camel}`] as well.
pub struct {camel}Pre<T: 'static> {{
    instance_pre: {wt}::component::InstancePre<T>,
    indices: {camel}Indices,
}}

impl<T: 'static> Clone for {camel}Pre<T> {{
    fn clone(&self) -> Self {{
        Self {{
            instance_pre: self.instance_pre.clone(),
            indices: self.indices.clone(),
        }}
    }}
}}

impl<_T: 'static> {camel}Pre<_T> {{
    /// Creates a new copy of `{camel}Pre` bindings which can then
    /// be used to instantiate into a particular store.
    ///
    /// This method may fail if the component behind `instance_pre`
    /// does not have the required exports.
    pub fn new(instance_pre: {wt}::component::InstancePre<_T>) -> {wt}::Result<Self> {{
        let indices = {camel}Indices::new(&instance_pre)?;
        Ok(Self {{ instance_pre, indices }})
    }}

    pub fn engine(&self) -> &{wt}::Engine {{
        self.instance_pre.engine()
    }}

    pub fn instance_pre(&self) -> &{wt}::component::InstancePre<_T> {{
        &self.instance_pre
    }}

    /// Instantiates a new instance of [`{camel}`] within the
    /// `store` provided.
    ///
    /// This function will use `self` as the pre-instantiated
    /// instance to perform instantiation. Afterwards the preloaded
    /// indices in `self` are used to lookup all exports on the
    /// resulting instance.
    pub fn instantiate(
        &self,
        mut store: impl {wt}::AsContextMut<Data = _T>,
    ) -> {wt}::Result<{camel}> {{
        let mut store = store.as_context_mut();
        let instance = self.instance_pre.instantiate(&mut store)?;
        self.indices.load(&mut store, &instance)
    }}
}}
"
        );

        if cfg!(feature = "async") {
            uwriteln!(
                self.src,
                "
impl<_T: Send + 'static> {camel}Pre<_T> {{
    /// Same as [`Self::instantiate`], except with `async`.
    pub async fn instantiate_async(
        &self,
        mut store: impl {wt}::AsContextMut<Data = _T>,
    ) -> {wt}::Result<{camel}> {{
        let mut store = store.as_context_mut();
        let instance = self.instance_pre.instantiate_async(&mut store).await?;
        self.indices.load(&mut store, &instance)
    }}
}}
"
            );
        }

        uwriteln!(
            self.src,
            "
            /// Auto-generated bindings for index of the exports of
            /// `{world_name}`.
            ///
            /// This is an implementation detail of [`{camel}Pre`] and can
            /// be constructed if needed as well.
            ///
            /// For more information see [`{camel}`] as well.
            #[derive(Clone)]
            pub struct {camel}Indices {{"
        );
        for (name, field) in self.exports.fields.iter() {
            uwriteln!(self.src, "{name}: {},", field.ty_index);
        }
        self.src.push_str("}\n");

        uwriteln!(
            self.src,
            "
                /// Auto-generated bindings for an instance a component which
                /// implements the world `{world_name}`.
                ///
                /// This structure can be created through a number of means
                /// depending on your requirements and what you have on hand:
                ///
                /// * The most convenient way is to use
                ///   [`{camel}::instantiate`] which only needs a
                ///   [`Store`], [`Component`], and [`Linker`].
                ///
                /// * Alternatively you can create a [`{camel}Pre`] ahead of
                ///   time with a [`Component`] to front-load string lookups
                ///   of exports once instead of per-instantiation. This
                ///   method then uses [`{camel}Pre::instantiate`] to
                ///   create a [`{camel}`].
                ///
                /// * If you've instantiated the instance yourself already
                ///   then you can use [`{camel}::new`].
                ///
                /// These methods are all equivalent to one another and move
                /// around the tradeoff of what work is performed when.
                ///
                /// [`Store`]: {wt}::Store
                /// [`Component`]: {wt}::component::Component
                /// [`Linker`]: {wt}::component::Linker
                pub struct {camel} {{"
        );
        for (name, field) in self.exports.fields.iter() {
            uwriteln!(self.src, "{name}: {},", field.ty);
        }
        self.src.push_str("}\n");

        let world_trait = self.world_imports_trait(resolve, world);

        uwriteln!(self.src, "const _: () = {{");

        uwriteln!(
            self.src,
            "impl {camel}Indices {{
                /// Creates a new copy of `{camel}Indices` bindings which can then
                /// be used to instantiate into a particular store.
                ///
                /// This method may fail if the component does not have the
                /// required exports.
                pub fn new<_T>(_instance_pre: &{wt}::component::InstancePre<_T>) -> {wt}::Result<Self> {{
                    let _component = _instance_pre.component();
                    let _instance_type = _instance_pre.instance_type();
            ",
        );
        for (name, field) in self.exports.fields.iter() {
            uwriteln!(self.src, "let {name} = {};", field.get_index);
        }
        uwriteln!(self.src, "Ok({camel}Indices {{");
        for (name, _) in self.exports.fields.iter() {
            uwriteln!(self.src, "{name},");
        }
        uwriteln!(self.src, "}})");
        uwriteln!(self.src, "}}"); // close `fn new`

        uwriteln!(
            self.src,
            "
                /// Uses the indices stored in `self` to load an instance
                /// of [`{camel}`] from the instance provided.
                ///
                /// Note that at this time this method will additionally
                /// perform type-checks of all exports.
                pub fn load(
                    &self,
                    mut store: impl {wt}::AsContextMut,
                    instance: &{wt}::component::Instance,
                ) -> {wt}::Result<{camel}> {{
                    let _ = &mut store;
                    let _instance = instance;
            ",
        );
        for (name, field) in self.exports.fields.iter() {
            uwriteln!(self.src, "let {name} = {};", field.load);
        }
        uwriteln!(self.src, "Ok({camel} {{");
        for (name, _) in self.exports.fields.iter() {
            uwriteln!(self.src, "{name},");
        }
        uwriteln!(self.src, "}})");
        uwriteln!(self.src, "}}"); // close `fn load`
        uwriteln!(self.src, "}}"); // close `impl {camel}Indices`

        uwriteln!(
            self.src,
            "impl {camel} {{
                /// Convenience wrapper around [`{camel}Pre::new`] and
                /// [`{camel}Pre::instantiate`].
                pub fn instantiate<_T>(
                    store: impl {wt}::AsContextMut<Data = _T>,
                    component: &{wt}::component::Component,
                    linker: &{wt}::component::Linker<_T>,
                ) -> {wt}::Result<{camel}> {{
                    let pre = linker.instantiate_pre(component)?;
                    {camel}Pre::new(pre)?.instantiate(store)
                }}

                /// Convenience wrapper around [`{camel}Indices::new`] and
                /// [`{camel}Indices::load`].
                pub fn new(
                    mut store: impl {wt}::AsContextMut,
                    instance: &{wt}::component::Instance,
                ) -> {wt}::Result<{camel}> {{
                    let indices = {camel}Indices::new(&instance.instance_pre(&store))?;
                    indices.load(&mut store, instance)
                }}
            ",
        );

        if cfg!(feature = "async") {
            uwriteln!(
                self.src,
                "
                    /// Convenience wrapper around [`{camel}Pre::new`] and
                    /// [`{camel}Pre::instantiate_async`].
                    pub async fn instantiate_async<_T>(
                        store: impl {wt}::AsContextMut<Data = _T>,
                        component: &{wt}::component::Component,
                        linker: &{wt}::component::Linker<_T>,
                    ) -> {wt}::Result<{camel}>
                        where _T: Send,
                    {{
                        let pre = linker.instantiate_pre(component)?;
                        {camel}Pre::new(pre)?.instantiate_async(store).await
                    }}
                ",
            );
        }
        self.world_add_to_linker(resolve, world, world_trait.as_ref());

        for func in self.exports.funcs.iter() {
            self.src.push_str(func);
        }

        uwriteln!(self.src, "}}"); // close `impl {camel}`

        uwriteln!(self.src, "}};"); // close `const _: () = ...
    }

    fn finish(&mut self, resolve: &Resolve, world: WorldId) -> anyhow::Result<String> {
        let remapping_keys = self.opts.with.keys().cloned().collect::<HashSet<String>>();

        let mut unused_keys = remapping_keys
            .difference(&self.used_with_opts)
            .map(|s| s.as_str())
            .collect::<Vec<&str>>();

        unused_keys.sort();

        if !unused_keys.is_empty() {
            anyhow::bail!(
                "interfaces were specified in the `with` config option but are not referenced in the target world: {unused_keys:?}"
            );
        }

        if !self.opts.only_interfaces {
            self.build_world_struct(resolve, world)
        }

        self.opts.imports.assert_all_rules_used("imports")?;
        self.opts.exports.assert_all_rules_used("exports")?;

        let imports = mem::take(&mut self.import_interfaces);
        self.emit_modules(
            imports
                .into_iter()
                .map(|(_, i)| (i.contents, i.name))
                .collect(),
        );

        let exports = mem::take(&mut self.exports.modules);
        self.emit_modules(exports);

        let named_imports = mem::take(&mut self.named_import_modules);
        self.emit_modules(named_imports);

        let mut src = mem::take(&mut self.src);
        if self.opts.rustfmt {
            let mut child = Command::new("rustfmt")
                .arg("--edition=2018")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .spawn()
                .expect("failed to spawn `rustfmt`");
            child
                .stdin
                .take()
                .unwrap()
                .write_all(src.as_bytes())
                .unwrap();
            src.as_mut_string().truncate(0);
            child
                .stdout
                .take()
                .unwrap()
                .read_to_string(src.as_mut_string())
                .unwrap();
            let status = child.wait().unwrap();
            assert!(status.success());
        }

        Ok(src.into())
    }

    fn emit_modules(&mut self, modules: Vec<(String, InterfaceName)>) {
        #[derive(Default)]
        struct Module {
            submodules: BTreeMap<String, Module>,
            contents: Vec<String>,
        }
        let mut map = Module::default();
        for (module, name) in modules {
            let path = match name {
                InterfaceName::Remapped { local_path, .. } => local_path,
                InterfaceName::Path(path) => path,
            };
            let mut cur = &mut map;
            for name in path[..path.len() - 1].iter() {
                cur = cur
                    .submodules
                    .entry(name.clone())
                    .or_insert(Module::default());
            }
            cur.contents.push(module);
        }

        emit(&mut self.src, map);

        fn emit(me: &mut Source, module: Module) {
            for (name, submodule) in module.submodules {
                uwriteln!(me, "pub mod {name} {{");
                emit(me, submodule);
                uwriteln!(me, "}}");
            }
            for submodule in module.contents {
                uwriteln!(me, "{submodule}");
            }
        }
    }

    /// Attempts to find the `key`, possibly with the resource projection
    /// `item`, within the `with` map provided to bindings configuration.
    fn lookup_replacement(
        &mut self,
        resolve: &Resolve,
        key: &WorldKey,
        item: Option<&str>,
    ) -> Option<String> {
        let item = match item {
            Some(item) => LookupItem::Name(item),
            None => LookupItem::None,
        };

        for (lookup, mut projection) in lookup_keys(resolve, key, item) {
            if let Some(renamed) = self.opts.with.get(&lookup) {
                projection.push(renamed.clone());
                projection.reverse();
                self.used_with_opts.insert(lookup);
                return Some(projection.join("::"));
            }
        }

        None
    }

    fn wasmtime_path(&self) -> String {
        self.opts
            .wasmtime_crate
            .clone()
            .unwrap_or("wasmtime".to_string())
    }
}

enum LookupItem<'a> {
    None,
    Name(&'a str),
    InterfaceNoPop,
}

fn lookup_keys(
    resolve: &Resolve,
    key: &WorldKey,
    item: LookupItem<'_>,
) -> Vec<(String, Vec<String>)> {
    struct Name<'a> {
        prefix: Prefix,
        item: Option<&'a str>,
    }

    #[derive(Copy, Clone)]
    enum Prefix {
        Namespace(PackageId),
        UnversionedPackage(PackageId),
        VersionedPackage(PackageId),
        UnversionedInterface(InterfaceId),
        VersionedInterface(InterfaceId),
    }

    let prefix = match key {
        WorldKey::Interface(id) => Prefix::VersionedInterface(*id),

        // Non-interface-keyed names don't get the lookup logic below,
        // they're relatively uncommon so only lookup the precise key here.
        WorldKey::Name(key) => {
            let to_lookup = match item {
                LookupItem::Name(item) => format!("{key}.{item}"),
                LookupItem::None | LookupItem::InterfaceNoPop => key.to_string(),
            };
            return vec![(to_lookup, Vec::new())];
        }
    };

    // Here names are iteratively attempted as `key` + `item` is "walked to
    // its root" and each attempt is consulted in `self.opts.with`. This
    // loop will start at the leaf, the most specific path, and then walk to
    // the root, popping items, trying to find a result.
    //
    // Each time a name is "popped" the projection from the next path is
    // pushed onto `projection`. This means that if we actually find a match
    // then `projection` is a collection of namespaces that results in the
    // final replacement name.
    let (interface_required, item) = match item {
        LookupItem::None => (false, None),
        LookupItem::Name(s) => (false, Some(s)),
        LookupItem::InterfaceNoPop => (true, None),
    };
    let mut name = Name { prefix, item };
    let mut projection = Vec::new();
    let mut ret = Vec::new();
    loop {
        let lookup = name.lookup_key(resolve);
        ret.push((lookup, projection.clone()));
        if !name.pop(resolve, &mut projection) {
            break;
        }
        if interface_required {
            match name.prefix {
                Prefix::VersionedInterface(_) | Prefix::UnversionedInterface(_) => {}
                _ => break,
            }
        }
    }

    return ret;

    impl<'a> Name<'a> {
        fn lookup_key(&self, resolve: &Resolve) -> String {
            let mut s = self.prefix.lookup_key(resolve);
            if let Some(item) = self.item {
                s.push_str(".");
                s.push_str(item);
            }
            s
        }

        fn pop(&mut self, resolve: &'a Resolve, projection: &mut Vec<String>) -> bool {
            match (self.item, self.prefix) {
                // If this is a versioned resource name, try the unversioned
                // resource name next.
                (Some(_), Prefix::VersionedInterface(id)) => {
                    self.prefix = Prefix::UnversionedInterface(id);
                    true
                }
                // If this is an unversioned resource name then time to
                // ignore the resource itself and move on to the next most
                // specific item, versioned interface names.
                (Some(item), Prefix::UnversionedInterface(id)) => {
                    self.prefix = Prefix::VersionedInterface(id);
                    self.item = None;
                    projection.push(item.to_upper_camel_case());
                    true
                }
                (Some(_), _) => unreachable!(),
                (None, _) => self.prefix.pop(resolve, projection),
            }
        }
    }

    impl Prefix {
        fn lookup_key(&self, resolve: &Resolve) -> String {
            match *self {
                Prefix::Namespace(id) => resolve.packages[id].name.namespace.clone(),
                Prefix::UnversionedPackage(id) => {
                    let mut name = resolve.packages[id].name.clone();
                    name.version = None;
                    name.to_string()
                }
                Prefix::VersionedPackage(id) => resolve.packages[id].name.to_string(),
                Prefix::UnversionedInterface(id) => {
                    let id = resolve.id_of(id).unwrap();
                    match id.find('@') {
                        Some(i) => id[..i].to_string(),
                        None => id,
                    }
                }
                Prefix::VersionedInterface(id) => resolve.id_of(id).unwrap(),
            }
        }

        fn pop(&mut self, resolve: &Resolve, projection: &mut Vec<String>) -> bool {
            *self = match *self {
                // try the unversioned interface next
                Prefix::VersionedInterface(id) => Prefix::UnversionedInterface(id),
                // try this interface's versioned package next
                Prefix::UnversionedInterface(id) => {
                    let iface = &resolve.interfaces[id];
                    let name = iface.name.as_ref().unwrap();
                    projection.push(to_rust_ident(name));
                    Prefix::VersionedPackage(iface.package.unwrap())
                }
                // try the unversioned package next
                Prefix::VersionedPackage(id) => Prefix::UnversionedPackage(id),
                // try this package's namespace next
                Prefix::UnversionedPackage(id) => {
                    let name = &resolve.packages[id].name;
                    projection.push(to_rust_ident(&name.name));
                    Prefix::Namespace(id)
                }
                // nothing left to try any more
                Prefix::Namespace(_) => return false,
            };
            true
        }
    }
}

impl Wasmtime {
    fn has_world_imports_trait(&self, resolve: &Resolve, world: WorldId) -> bool {
        !self.world_import_functions.is_empty() || get_world_resources(resolve, world).count() > 0
    }

    fn world_imports_trait(&mut self, resolve: &Resolve, world: WorldId) -> Option<GeneratedTrait> {
        if !self.has_world_imports_trait(resolve, world) {
            return None;
        }

        let world_camel = to_rust_upper_camel_case(&resolve.worlds[world].name);

        let functions = self.world_import_functions.clone();
        let mut generator = InterfaceGenerator::new(self, resolve);
        let generated_trait = generator.generate_trait(
            &format!("{world_camel}Imports"),
            &functions
                .iter()
                .filter(|f| f.kind.resource().is_none())
                .collect::<Vec<_>>(),
            &[],
            &get_world_resources(resolve, world).collect::<Vec<_>>(),
        );
        let src = String::from(mem::take(&mut generator.src));
        self.src.push_str(&src);
        Some(generated_trait)
    }

    fn import_interface_paths(&self) -> Vec<(InterfaceId, String, Option<String>)> {
        self.import_interfaces
            .iter()
            .filter(|(_, i)| i.in_world)
            .map(|(id, _)| (*id, None))
            .chain(
                self.world_implements_interfaces
                    .iter()
                    .map(|(name, id)| (*id, Some(name.clone()))),
            )
            .map(|(id, name_override)| (id, self.import_interface_path(&id), name_override))
            .collect()
    }

    fn import_interface_path(&self, id: &InterfaceId) -> String {
        match &self.interface_names[id] {
            InterfaceName::Path(path) => path.join("::"),
            InterfaceName::Remapped { name_at_root, .. } => name_at_root.clone(),
        }
    }

    fn import_interface_all_func_flags(&self, id: InterfaceId) -> FunctionFlags {
        self.import_interfaces[&id].all_func_flags
    }

    fn world_host_traits(
        &self,
        world_trait: Option<&GeneratedTrait>,
    ) -> (Vec<String>, Vec<String>) {
        let mut without_store = Vec::new();
        let mut without_store_async = false;
        let mut with_store = Vec::new();
        let mut with_store_async = false;
        for (id, path, _) in self.import_interface_paths() {
            without_store.push(format!("{path}::Host"));
            let flags = self.import_interface_all_func_flags(id);
            without_store_async = without_store_async || flags.contains(FunctionFlags::ASYNC);

            // Note that the requirement of `HostWithStore` is technically
            // dependent on `FunctionFlags::STORE`, but when `with` is in use we
            // don't necessarily know whether the other bindings generation
            // specified this flag or not. To handle that always assume that a
            // `HostWithStore` bound is needed.
            with_store.push(format!("{path}::HostWithStore<T>"));
            with_store_async = with_store_async || flags.contains(FunctionFlags::ASYNC);
        }
        if let Some(world_trait) = world_trait {
            without_store.push(world_trait.name.clone());
            without_store_async =
                without_store_async || world_trait.all_func_flags.contains(FunctionFlags::ASYNC);

            if world_trait.with_store_name.is_some() {
                with_store.extend(world_trait.with_store_name.clone());
                with_store_async =
                    with_store_async || world_trait.all_func_flags.contains(FunctionFlags::ASYNC);
            }
        }
        if without_store_async {
            without_store.push("Send".to_string());
        }
        if with_store_async {
            with_store.push("Send".to_string());
        }
        (without_store, with_store)
    }

    fn world_add_to_linker(
        &mut self,
        resolve: &Resolve,
        world: WorldId,
        world_trait: Option<&GeneratedTrait>,
    ) {
        let has_world_imports_trait = self.has_world_imports_trait(resolve, world);
        if self.import_interfaces.is_empty() && !has_world_imports_trait {
            return;
        }

        let (options_param, options_arg) = if self.world_link_options.has_any() {
            ("options: &LinkOptions,", ", options")
        } else {
            ("", "")
        };

        let mut all_func_flags = FunctionFlags::empty();
        if let Some(world_trait) = world_trait {
            all_func_flags |= world_trait.all_func_flags;
        }
        for i in self.import_interfaces.values() {
            all_func_flags |= i.all_func_flags;
        }

        all_func_flags |= self.opts.imports.default;
        all_func_flags |= self.opts.exports.default;

        let opt_t_send_bound =
            if all_func_flags.contains(FunctionFlags::ASYNC) || self.opts.require_store_data_send {
                "+ Send"
            } else {
                ""
            };

        let wt = self.wasmtime_path();
        if let Some(world_trait) = world_trait {
            let d_bound = match &world_trait.with_store_name {
                Some(name) => name.clone(),
                None => format!("{wt}::component::HasData"),
            };
            uwrite!(
                self.src,
                "
                    pub fn add_to_linker_imports<T, D>(
                        linker: &mut {wt}::component::Linker<T>,
                        {options_param}
                        host_getter: fn(&mut T) -> D::Data<'_>,
                    ) -> {wt}::Result<()>
                        where
                            D: {d_bound},
                            for<'a> D::Data<'a>: {name},
                            T: 'static {opt_t_send_bound}
                    {{
                        let mut linker = linker.root();
                ",
                name = world_trait.name,
            );
            let gate = FeatureGate::open(&mut self.src, &resolve.worlds[world].stability);
            for (ty, _name) in get_world_resources(resolve, world) {
                self.generate_add_resource_to_linker(None, None, "linker", resolve, ty);
            }
            for f in self.world_import_functions.clone() {
                let mut generator = InterfaceGenerator::new(self, resolve);
                generator.generate_add_function_to_linker(TypeOwner::World(world), &f, "linker");
                let src = String::from(generator.src);
                self.src.push_str(&src);
                self.src.push_str("\n");
            }
            gate.close(&mut self.src);
            uwriteln!(self.src, "Ok(())\n}}");
        }

        let (sync_bounds, concurrent_bounds) = self.world_host_traits(world_trait);
        let sync_bounds = sync_bounds.join(" + ");
        let concurrent_bounds = concurrent_bounds.join(" + ");
        let d_bounds = if !concurrent_bounds.is_empty() {
            concurrent_bounds
        } else {
            format!("{wt}::component::HasData")
        };

        uwriteln!(
            self.src,
            "
                pub fn add_to_linker<T, D>(
                    linker: &mut {wt}::component::Linker<T>,
                    {options_param}
                    host_getter: fn(&mut T) -> D::Data<'_>,
                ) -> {wt}::Result<()>
                    where
                        D: {d_bounds},
                        for<'a> D::Data<'a>: {sync_bounds},
                        T: 'static {opt_t_send_bound}
                {{
            "
        );
        let gate = FeatureGate::open(&mut self.src, &resolve.worlds[world].stability);
        if has_world_imports_trait {
            uwriteln!(
                self.src,
                "Self::add_to_linker_imports::<T, D>(linker {options_arg}, host_getter)?;"
            );
        }
        for (interface_id, path, name_override) in self.import_interface_paths() {
            let options_arg = if self.interface_link_options[&interface_id].has_any() {
                ", &options.into()"
            } else {
                ""
            };

            let import_stability = resolve.worlds[world]
                .imports
                .iter()
                .filter_map(|(_, i)| match i {
                    WorldItem::Interface { id, stability, .. } if *id == interface_id => {
                        Some(stability.clone())
                    }
                    _ => None,
                })
                .next()
                .unwrap_or(Stability::Unknown);

            let gate = FeatureGate::open(&mut self.src, &import_stability);
            match &name_override {
                Some(name) => {
                    uwriteln!(
                        self.src,
                        "{path}::add_to_linker_instance::<T, D>(
                            &mut linker.instance({name:?})?
                            {options_arg},
                            host_getter,
                        )?;"
                    );
                }
                None => {
                    uwriteln!(
                        self.src,
                        "{path}::add_to_linker::<T, D>(
                            linker {options_arg}, host_getter,
                        )?;"
                    );
                }
            }
            gate.close(&mut self.src);
        }
        gate.close(&mut self.src);
        uwriteln!(self.src, "Ok(())\n}}");
    }

    fn generate_add_resource_to_linker(
        &mut self,
        key: Option<&WorldKey>,
        src: Option<&mut Source>,
        inst: &str,
        resolve: &Resolve,
        ty: TypeId,
    ) {
        let ty = &resolve.types[ty];
        let name = ty.name.as_ref().unwrap();
        let stability = &ty.stability;
        let wt = self.wasmtime_path();
        let src = src.unwrap_or(&mut self.src);
        let gate = FeatureGate::open(src, stability);
        let camel = name.to_upper_camel_case();

        let flags = self.opts.imports.resource_drop_flags(resolve, key, name);
        if flags.contains(FunctionFlags::ASYNC) {
            if flags.contains(FunctionFlags::STORE) {
                uwriteln!(
                    src,
                    "{inst}.resource_concurrent(
                        \"{name}\",
                        {wt}::component::ResourceType::host::<{camel}>(),
                        move |caller: &{wt}::component::Accessor::<T>, rep| {{
                            {wt}::component::__internal::Box::pin(async move {{
                                let accessor = &caller.with_getter(host_getter);
                                {wt}::ToWasmtimeResult::to_wasmtime_result(
                                    Host{camel}WithStore::<T>::drop(accessor, {wt}::component::Resource::new_own(rep)).await
                                )
                            }})
                        }},
                    )?;"
                )
            } else {
                uwriteln!(
                    src,
                    "{inst}.resource_async(
                        \"{name}\",
                        {wt}::component::ResourceType::host::<{camel}>(),
                        move |mut store, rep| {{
                            {wt}::component::__internal::Box::new(async move {{
                                {wt}::ToWasmtimeResult::to_wasmtime_result(
                                    Host{camel}::drop(&mut host_getter(store.data_mut()), {wt}::component::Resource::new_own(rep)).await
                                )
                            }})
                        }},
                    )?;"
                )
            }
        } else {
            let (first_arg, trait_suffix) = if flags.contains(FunctionFlags::STORE) {
                (
                    format!("{wt}::component::Access::new(store, host_getter)"),
                    "WithStore::<T>",
                )
            } else {
                ("&mut host_getter(store.data_mut())".to_string(), "")
            };
            uwriteln!(
                src,
                "{inst}.resource(
                    \"{name}\",
                    {wt}::component::ResourceType::host::<{camel}>(),
                    move |mut store, rep| -> {wt}::Result<()> {{

                        let resource = {wt}::component::Resource::new_own(rep);
                        {wt}::ToWasmtimeResult::to_wasmtime_result(
                            Host{camel}{trait_suffix}::drop({first_arg}, resource)
                        )
                    }},
                )?;",
            )
        }
        gate.close(src);
    }
}

struct InterfaceGenerator<'a> {
    src: Source,
    generator: &'a mut Wasmtime,
    resolve: &'a Resolve,
    current_interface: Option<(InterfaceId, &'a WorldKey, InterfaceKind)>,
    all_func_flags: FunctionFlags,

    /// The type that represents the embedder-chosen "id" for named imports.
    named_import_id: Option<String>,
}

impl<'a> InterfaceGenerator<'a> {
    fn new(generator: &'a mut Wasmtime, resolve: &'a Resolve) -> InterfaceGenerator<'a> {
        InterfaceGenerator {
            src: Source::default(),
            generator,
            resolve,
            current_interface: None,
            all_func_flags: FunctionFlags::empty(),
            named_import_id: None,
        }
    }

    fn types_imported(&self) -> bool {
        match self.current_interface {
            Some((_, _, InterfaceKind::Export)) => false,
            _ => true,
        }
    }

    fn types(&mut self, id: InterfaceId) {
        for (name, id) in self.resolve.interfaces[id].types.iter() {
            self.define_type(name, *id);
        }
    }

    fn define_type(&mut self, name: &str, id: TypeId) {
        let ty = &self.resolve.types[id];
        match &ty.kind {
            TypeDefKind::Record(record) => self.type_record(id, name, record, &ty.docs),
            TypeDefKind::Flags(flags) => self.type_flags(id, name, flags, &ty.docs),
            TypeDefKind::Tuple(tuple) => self.type_tuple(id, name, tuple, &ty.docs),
            TypeDefKind::Enum(enum_) => self.type_enum(id, name, enum_, &ty.docs),
            TypeDefKind::Variant(variant) => self.type_variant(id, name, variant, &ty.docs),
            TypeDefKind::Option(t) => self.type_option(id, name, t, &ty.docs),
            TypeDefKind::Result(r) => self.type_result(id, name, r, &ty.docs),
            TypeDefKind::List(t) => self.type_list(id, name, t, &ty.docs),
            TypeDefKind::Type(t) => self.type_alias(id, name, t, &ty.docs),
            TypeDefKind::Future(t) => self.type_future(id, name, t.as_ref(), &ty.docs),
            TypeDefKind::Stream(t) => self.type_stream(id, name, t.as_ref(), &ty.docs),
            TypeDefKind::Handle(handle) => self.type_handle(id, name, handle, &ty.docs),
            TypeDefKind::Resource => self.type_resource(id, name, ty, &ty.docs),
            TypeDefKind::Map(k, v) => self.type_map(id, name, k, v, &ty.docs),
            TypeDefKind::Unknown => unreachable!(),
            TypeDefKind::FixedLengthList(..) => todo!(),
        }
    }

    fn type_handle(&mut self, id: TypeId, name: &str, handle: &Handle, docs: &Docs) {
        self.rustdoc(docs);
        let name = name.to_upper_camel_case();
        uwriteln!(self.src, "pub type {name} = ");
        self.print_handle(handle);
        self.push_str(";\n");
        self.assert_type(id, &name);
    }

    fn type_resource(&mut self, id: TypeId, name: &str, _resource: &TypeDef, docs: &Docs) {
        let camel = name.to_upper_camel_case();
        let wt = self.generator.wasmtime_path();

        if self.types_imported() {
            self.rustdoc(docs);

            let replacement = match self.current_interface {
                Some((_, key, _)) => {
                    self.generator
                        .lookup_replacement(self.resolve, key, Some(name))
                }
                None => {
                    self.generator.used_with_opts.insert(name.into());
                    self.generator.opts.with.get(name).cloned()
                }
            };
            match replacement {
                Some(path) => {
                    uwriteln!(
                        self.src,
                        "pub use {}{path} as {camel};",
                        self.path_to_root()
                    );
                }
                None => {
                    uwriteln!(self.src, "pub enum {camel} {{}}");
                }
            }

            // Generate resource trait

            let functions = get_resource_functions(self.resolve, id);
            let trait_ = self.generate_trait(
                &format!("Host{camel}"),
                &functions,
                &[ExtraTraitMethod::ResourceDrop { name }],
                &[],
            );
            self.all_func_flags |= trait_.all_func_flags;
        } else {
            self.rustdoc(docs);
            uwriteln!(
                self.src,
                "
                    pub type {camel} = {wt}::component::ResourceAny;

                    pub struct Guest{camel}<'a> {{
                        funcs: &'a Guest,
                    }}
                "
            );
        }
    }

    fn type_record(&mut self, id: TypeId, _name: &str, record: &Record, docs: &Docs) {
        let info = self.info(id);
        let wt = self.generator.wasmtime_path();

        // We use a BTree set to make sure we don't have any duplicates and we have a stable order
        let additional_derives: BTreeSet<String> = self
            .generator
            .opts
            .additional_derive_attributes
            .iter()
            .cloned()
            .collect();

        for (name, mode) in self.modes_of(id) {
            let lt = self.lifetime_for(&info, mode);
            self.rustdoc(docs);

            let mut derives = additional_derives.clone();

            uwriteln!(self.src, "#[derive({wt}::component::ComponentType)]");
            if lt.is_none() {
                uwriteln!(self.src, "#[derive({wt}::component::Lift)]");
            }
            uwriteln!(self.src, "#[derive({wt}::component::Lower)]");
            self.push_str("#[component(record)]\n");
            if let Some(path) = &self.generator.opts.wasmtime_crate {
                uwriteln!(self.src, "#[component(wasmtime_crate = {path})]\n");
            }

            if info.is_copy() {
                derives.extend(["Copy", "Clone"].into_iter().map(|s| s.to_string()));
            } else if info.is_clone() {
                derives.insert("Clone".to_string());
            }

            if !derives.is_empty() {
                self.push_str("#[derive(");
                self.push_str(&derives.into_iter().collect::<Vec<_>>().join(", "));
                self.push_str(")]\n")
            }

            self.push_str(&format!("pub struct {name}"));
            self.print_generics(lt);
            self.push_str(" {\n");
            for field in record.fields.iter() {
                self.rustdoc(&field.docs);
                self.push_str(&format!("#[component(name = \"{}\")]\n", field.name));
                self.push_str("pub ");
                self.push_str(&to_rust_ident(&field.name));
                self.push_str(": ");
                self.print_ty(&field.ty, mode);
                self.push_str(",\n");
            }
            self.push_str("}\n");

            self.push_str("impl");
            self.print_generics(lt);
            self.push_str(" core::fmt::Debug for ");
            self.push_str(&name);
            self.print_generics(lt);
            self.push_str(" {\n");
            self.push_str(
                "fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n",
            );
            self.push_str(&format!("f.debug_struct(\"{name}\")"));
            for field in record.fields.iter() {
                self.push_str(&format!(
                    ".field(\"{}\", &self.{})",
                    field.name,
                    to_rust_ident(&field.name)
                ));
            }
            self.push_str(".finish()\n");
            self.push_str("}\n");
            self.push_str("}\n");

            if info.error {
                self.push_str("impl");
                self.print_generics(lt);
                self.push_str(" core::fmt::Display for ");
                self.push_str(&name);
                self.print_generics(lt);
                self.push_str(" {\n");
                self.push_str(
                    "fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n",
                );
                self.push_str("write!(f, \"{:?}\", self)\n");
                self.push_str("}\n");
                self.push_str("}\n");

                self.push_str("impl core::error::Error for ");
                self.push_str(&name);
                self.push_str("{}\n");
            }
            self.assert_type(id, &name);
        }
    }

    fn type_tuple(&mut self, id: TypeId, _name: &str, tuple: &Tuple, docs: &Docs) {
        let info = self.info(id);
        for (name, mode) in self.modes_of(id) {
            let lt = self.lifetime_for(&info, mode);
            self.rustdoc(docs);
            self.push_str(&format!("pub type {name}"));
            self.print_generics(lt);
            self.push_str(" = (");
            for ty in tuple.types.iter() {
                self.print_ty(ty, mode);
                self.push_str(",");
            }
            self.push_str(");\n");
            self.assert_type(id, &name);
        }
    }

    fn type_flags(&mut self, id: TypeId, name: &str, flags: &Flags, docs: &Docs) {
        self.rustdoc(docs);
        let wt = self.generator.wasmtime_path();
        let rust_name = to_rust_upper_camel_case(name);
        uwriteln!(self.src, "{wt}::component::flags!(\n");
        self.src.push_str(&format!("{rust_name} {{\n"));
        for flag in flags.flags.iter() {
            // TODO wasmtime-component-macro doesn't support docs for flags rn
            uwrite!(
                self.src,
                "#[component(name=\"{}\")] const {};\n",
                flag.name,
                flag.name.to_shouty_snake_case()
            );
        }
        self.src.push_str("}\n");
        self.src.push_str(");\n\n");
        self.assert_type(id, &rust_name);
    }

    fn type_variant(&mut self, id: TypeId, _name: &str, variant: &Variant, docs: &Docs) {
        self.print_rust_enum(
            id,
            variant.cases.iter().map(|c| {
                (
                    c.name.to_upper_camel_case(),
                    Some(c.name.clone()),
                    &c.docs,
                    c.ty.as_ref(),
                )
            }),
            docs,
            "variant",
        );
    }

    fn type_option(&mut self, id: TypeId, _name: &str, payload: &Type, docs: &Docs) {
        let info = self.info(id);

        for (name, mode) in self.modes_of(id) {
            self.rustdoc(docs);
            let lt = self.lifetime_for(&info, mode);
            self.push_str(&format!("pub type {name}"));
            self.print_generics(lt);
            self.push_str("= Option<");
            self.print_ty(payload, mode);
            self.push_str(">;\n");
            self.assert_type(id, &name);
        }
    }

    // Emit a double-check that the wit-parser-understood size of a type agrees
    // with the Wasmtime-understood size of a type.
    fn assert_type(&mut self, id: TypeId, name: &str) {
        self.push_str("const _: () = {\n");
        let wt = self.generator.wasmtime_path();
        uwriteln!(
            self.src,
            "assert!({} == <{name} as {wt}::component::ComponentType>::SIZE32);",
            self.generator.sizes.size(&Type::Id(id)).size_wasm32(),
        );
        uwriteln!(
            self.src,
            "assert!({} == <{name} as {wt}::component::ComponentType>::ALIGN32);",
            self.generator.sizes.align(&Type::Id(id)).align_wasm32(),
        );
        self.push_str("};\n");
    }

    fn print_rust_enum<'b>(
        &mut self,
        id: TypeId,
        cases: impl IntoIterator<Item = (String, Option<String>, &'b Docs, Option<&'b Type>)> + Clone,
        docs: &Docs,
        derive_component: &str,
    ) where
        Self: Sized,
    {
        let info = self.info(id);
        let wt = self.generator.wasmtime_path();

        // We use a BTree set to make sure we don't have any duplicates and we have a stable order
        let additional_derives: BTreeSet<String> = self
            .generator
            .opts
            .additional_derive_attributes
            .iter()
            .cloned()
            .collect();

        for (name, mode) in self.modes_of(id) {
            let name = to_rust_upper_camel_case(&name);

            let mut derives = additional_derives.clone();

            self.rustdoc(docs);
            let lt = self.lifetime_for(&info, mode);
            uwriteln!(self.src, "#[derive({wt}::component::ComponentType)]");
            if lt.is_none() {
                uwriteln!(self.src, "#[derive({wt}::component::Lift)]");
            }
            uwriteln!(self.src, "#[derive({wt}::component::Lower)]");
            self.push_str(&format!("#[component({derive_component})]\n"));
            if let Some(path) = &self.generator.opts.wasmtime_crate {
                uwriteln!(self.src, "#[component(wasmtime_crate = {path})]\n");
            }
            if info.is_copy() {
                derives.extend(["Copy", "Clone"].into_iter().map(|s| s.to_string()));
            } else if info.is_clone() {
                derives.insert("Clone".to_string());
            }

            if !derives.is_empty() {
                self.push_str("#[derive(");
                self.push_str(&derives.into_iter().collect::<Vec<_>>().join(", "));
                self.push_str(")]\n")
            }

            self.push_str(&format!("pub enum {name}"));
            self.print_generics(lt);
            self.push_str("{\n");
            for (case_name, component_name, docs, payload) in cases.clone() {
                self.rustdoc(docs);
                if let Some(n) = component_name {
                    self.push_str(&format!("#[component(name = \"{n}\")] "));
                }
                self.push_str(&case_name);
                if let Some(ty) = payload {
                    self.push_str("(");
                    self.print_ty(ty, mode);
                    self.push_str(")")
                }
                self.push_str(",\n");
            }
            self.push_str("}\n");

            self.print_rust_enum_debug(
                id,
                mode,
                &name,
                cases
                    .clone()
                    .into_iter()
                    .map(|(name, _attr, _docs, ty)| (name, ty)),
            );

            if info.error {
                self.push_str("impl");
                self.print_generics(lt);
                self.push_str(" core::fmt::Display for ");
                self.push_str(&name);
                self.print_generics(lt);
                self.push_str(" {\n");
                self.push_str(
                    "fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n",
                );
                self.push_str("write!(f, \"{:?}\", self)\n");
                self.push_str("}\n");
                self.push_str("}\n");

                self.push_str("impl");
                self.print_generics(lt);
                self.push_str(" core::error::Error for ");
                self.push_str(&name);
                self.print_generics(lt);
                self.push_str(" {}\n");
            }

            self.assert_type(id, &name);
        }
    }

    fn print_rust_enum_debug<'b>(
        &mut self,
        id: TypeId,
        mode: TypeMode,
        name: &str,
        cases: impl IntoIterator<Item = (String, Option<&'b Type>)>,
    ) where
        Self: Sized,
    {
        let info = self.info(id);
        let lt = self.lifetime_for(&info, mode);
        self.push_str("impl");
        self.print_generics(lt);
        self.push_str(" core::fmt::Debug for ");
        self.push_str(name);
        self.print_generics(lt);
        self.push_str(" {\n");
        self.push_str("fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n");
        self.push_str("match self {\n");
        for (case_name, payload) in cases {
            self.push_str(name);
            self.push_str("::");
            self.push_str(&case_name);
            if payload.is_some() {
                self.push_str("(e)");
            }
            self.push_str(" => {\n");
            self.push_str(&format!("f.debug_tuple(\"{name}::{case_name}\")"));
            if payload.is_some() {
                self.push_str(".field(e)");
            }
            self.push_str(".finish()\n");
            self.push_str("}\n");
        }
        self.push_str("}\n");
        self.push_str("}\n");
        self.push_str("}\n");
    }

    fn type_result(&mut self, id: TypeId, _name: &str, result: &Result_, docs: &Docs) {
        let info = self.info(id);

        for (name, mode) in self.modes_of(id) {
            self.rustdoc(docs);
            let lt = self.lifetime_for(&info, mode);
            self.push_str(&format!("pub type {name}"));
            self.print_generics(lt);
            self.push_str("= Result<");
            self.print_optional_ty(result.ok.as_ref(), mode);
            self.push_str(",");
            self.print_optional_ty(result.err.as_ref(), mode);
            self.push_str(">;\n");
            self.assert_type(id, &name);
        }
    }

    fn type_enum(&mut self, id: TypeId, name: &str, enum_: &Enum, docs: &Docs) {
        let info = self.info(id);
        let wt = self.generator.wasmtime_path();

        // We use a BTree set to make sure we don't have any duplicates and have a stable order
        let mut derives: BTreeSet<String> = self
            .generator
            .opts
            .additional_derive_attributes
            .iter()
            .cloned()
            .collect();

        derives.extend(
            ["Clone", "Copy", "PartialEq", "Eq"]
                .into_iter()
                .map(|s| s.to_string()),
        );

        let name = to_rust_upper_camel_case(name);
        self.rustdoc(docs);
        uwriteln!(self.src, "#[derive({wt}::component::ComponentType)]");
        uwriteln!(self.src, "#[derive({wt}::component::Lift)]");
        uwriteln!(self.src, "#[derive({wt}::component::Lower)]");
        self.push_str("#[component(enum)]\n");
        if let Some(path) = &self.generator.opts.wasmtime_crate {
            uwriteln!(self.src, "#[component(wasmtime_crate = {path})]\n");
        }

        self.push_str("#[derive(");
        self.push_str(&derives.into_iter().collect::<Vec<_>>().join(", "));
        self.push_str(")]\n");

        let repr = match enum_.cases.len().ilog2() {
            0..=7 => "u8",
            8..=15 => "u16",
            _ => "u32",
        };
        uwriteln!(self.src, "#[repr({repr})]");

        self.push_str(&format!("pub enum {name} {{\n"));
        for case in enum_.cases.iter() {
            self.rustdoc(&case.docs);
            self.push_str(&format!("#[component(name = \"{}\")]", case.name));
            self.push_str(&case.name.to_upper_camel_case());
            self.push_str(",\n");
        }
        self.push_str("}\n");

        // Auto-synthesize an implementation of the standard `Error` trait for
        // error-looking types based on their name.
        if info.error {
            self.push_str("impl ");
            self.push_str(&name);
            self.push_str("{\n");

            self.push_str("pub fn name(&self) -> &'static str {\n");
            self.push_str("match self {\n");
            for case in enum_.cases.iter() {
                self.push_str(&name);
                self.push_str("::");
                self.push_str(&case.name.to_upper_camel_case());
                self.push_str(" => \"");
                self.push_str(case.name.as_str());
                self.push_str("\",\n");
            }
            self.push_str("}\n");
            self.push_str("}\n");

            self.push_str("pub fn message(&self) -> &'static str {\n");
            self.push_str("match self {\n");
            for case in enum_.cases.iter() {
                self.push_str(&name);
                self.push_str("::");
                self.push_str(&case.name.to_upper_camel_case());
                self.push_str(" => \"");
                if let Some(contents) = &case.docs.contents {
                    self.push_str(contents.trim());
                }
                self.push_str("\",\n");
            }
            self.push_str("}\n");
            self.push_str("}\n");

            self.push_str("}\n");

            self.push_str("impl core::fmt::Debug for ");
            self.push_str(&name);
            self.push_str(
                "{\nfn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n",
            );
            self.push_str("f.debug_struct(\"");
            self.push_str(&name);
            self.push_str("\")\n");
            self.push_str(".field(\"code\", &(*self as i32))\n");
            self.push_str(".field(\"name\", &self.name())\n");
            self.push_str(".field(\"message\", &self.message())\n");
            self.push_str(".finish()\n");
            self.push_str("}\n");
            self.push_str("}\n");

            self.push_str("impl core::fmt::Display for ");
            self.push_str(&name);
            self.push_str(
                "{\nfn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {\n",
            );
            self.push_str("write!(f, \"{} (error {})\", self.name(), *self as i32)");
            self.push_str("}\n");
            self.push_str("}\n");
            self.push_str("\n");
            self.push_str("impl core::error::Error for ");
            self.push_str(&name);
            self.push_str("{}\n");
        } else {
            self.print_rust_enum_debug(
                id,
                TypeMode::Owned,
                &name,
                enum_
                    .cases
                    .iter()
                    .map(|c| (c.name.to_upper_camel_case(), None)),
            )
        }
        self.assert_type(id, &name);
    }

    fn type_alias(&mut self, id: TypeId, _name: &str, ty: &Type, docs: &Docs) {
        let info = self.info(id);
        for (name, mode) in self.modes_of(id) {
            self.rustdoc(docs);
            self.push_str(&format!("pub type {name}"));
            let lt = self.lifetime_for(&info, mode);
            self.print_generics(lt);
            self.push_str(" = ");
            self.print_ty(ty, mode);
            self.push_str(";\n");
            let def_id = resolve_type_definition_id(self.resolve, id);
            if !matches!(self.resolve().types[def_id].kind, TypeDefKind::Resource) {
                self.assert_type(id, &name);
            }
        }
    }

    fn type_list(&mut self, id: TypeId, _name: &str, ty: &Type, docs: &Docs) {
        let info = self.info(id);
        for (name, mode) in self.modes_of(id) {
            let lt = self.lifetime_for(&info, mode);
            self.rustdoc(docs);
            self.push_str(&format!("pub type {name}"));
            self.print_generics(lt);
            self.push_str(" = ");
            self.print_list(ty, mode);
            self.push_str(";\n");
            self.assert_type(id, &name);
        }
    }

    fn type_map(&mut self, id: TypeId, _name: &str, key: &Type, value: &Type, docs: &Docs) {
        let info = self.info(id);
        for (name, mode) in self.modes_of(id) {
            let lt = self.lifetime_for(&info, mode);
            self.rustdoc(docs);
            self.push_str(&format!("pub type {name}"));
            self.print_generics(lt);
            self.push_str(" = ");
            let key_ty = self.ty(key, mode);
            let value_ty = self.ty(value, mode);
            self.push_str(&format!("std::collections::HashMap<{key_ty}, {value_ty}>"));
            self.push_str(";\n");
            self.assert_type(id, &name);
        }
    }

    fn type_stream(&mut self, id: TypeId, name: &str, ty: Option<&Type>, docs: &Docs) {
        self.rustdoc(docs);
        self.push_str(&format!("pub type {name}"));
        self.print_generics(None);
        self.push_str(" = ");
        self.print_stream(ty);
        self.push_str(";\n");
        self.assert_type(id, &name);
    }

    fn type_future(&mut self, id: TypeId, name: &str, ty: Option<&Type>, docs: &Docs) {
        self.rustdoc(docs);
        self.push_str(&format!("pub type {name}"));
        self.print_generics(None);
        self.push_str(" = ");
        self.print_future(ty);
        self.push_str(";\n");
        self.assert_type(id, &name);
    }

    fn print_result_ty(&mut self, result: Option<Type>, mode: TypeMode) {
        match result {
            Some(ty) => self.print_ty(&ty, mode),
            None => self.push_str("()"),
        }
    }

    fn special_case_trappable_error(
        &mut self,
        func: &Function,
    ) -> Option<(&'a Result_, TypeId, String)> {
        let result = func.result?;

        // We fill in a special trappable error type in the case when a function has just one
        // result, which is itself a `result<a, e>`, and the `e` is *not* a primitive
        // (i.e. defined in std) type, and matches the typename given by the user.
        let id = match result {
            Type::Id(id) => id,
            _ => return None,
        };
        let result = match &self.resolve.types[id].kind {
            TypeDefKind::Result(r) => r,
            _ => return None,
        };
        let error_typeid = match result.err? {
            Type::Id(id) => resolve_type_definition_id(&self.resolve, id),
            _ => return None,
        };

        let name = self.generator.trappable_errors.get(&error_typeid)?;

        let mut path = self.path_to_root();
        uwrite!(path, "{name}");
        Some((result, error_typeid, path))
    }

    fn generate_add_to_linker(&mut self, id: InterfaceId, name: &str) {
        let iface = &self.resolve.interfaces[id];
        let owner = TypeOwner::Interface(id);
        let wt = self.generator.wasmtime_path();

        let mut required_conversion_traits = IndexSet::new();
        let extra_functions = {
            let mut functions = Vec::new();
            let mut errors_converted = IndexMap::new();
            let mut my_error_types = iface
                .types
                .iter()
                .filter(|(_, id)| self.generator.trappable_errors.contains_key(*id))
                .map(|(_, id)| *id)
                .collect::<Vec<_>>();
            my_error_types.extend(
                iface
                    .functions
                    .iter()
                    .filter_map(|(_, func)| self.special_case_trappable_error(func))
                    .map(|(_, id, _)| id),
            );
            for err_id in my_error_types {
                let err = &self.resolve.types[resolve_type_definition_id(self.resolve, err_id)];
                let err_name = err.name.as_ref().unwrap();
                let owner = match err.owner {
                    TypeOwner::Interface(i) => i,
                    _ => unimplemented!(),
                };
                match self.path_to_interface(owner) {
                    Some(path) => {
                        required_conversion_traits.insert(format!("{path}::Host"));
                    }
                    None => {
                        if errors_converted.insert(err_name, err_id).is_none() {
                            functions.push(ExtraTraitMethod::ErrorConvert {
                                name: err_name,
                                id: err_id,
                            })
                        }
                    }
                }
            }
            functions
        };

        // Generate the `pub trait` which represents the host functionality for
        // this import which additionally inherits from all resource traits
        // for this interface defined by `type_resource`.
        let generated_trait = self.generate_trait(
            "Host",
            &iface
                .functions
                .iter()
                .filter_map(|(_, f)| {
                    if f.kind.resource().is_none() {
                        Some(f)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
            &extra_functions,
            &get_resources(self.resolve, id).collect::<Vec<_>>(),
        );

        let opt_t_send_bound = if generated_trait
            .all_func_flags
            .contains(FunctionFlags::ASYNC)
        {
            "+ Send"
        } else {
            ""
        };

        let mut sync_bounds = "Host".to_string();

        for ty in required_conversion_traits {
            uwrite!(sync_bounds, " + {ty}");
        }

        let options_param = if self.generator.interface_link_options[&id].has_any() {
            "options: &LinkOptions,"
        } else {
            ""
        };
        let options_param_forward = if self.generator.interface_link_options[&id].has_any() {
            "options,"
        } else {
            ""
        };

        // For named imports the per-instance helper additionally accepts the
        // host-chosen `id` (by value, cloned into each closure).
        let id_param = match &self.named_import_id {
            Some(named) => format!("id: {named},"),
            None => String::new(),
        };

        uwriteln!(
            self.src,
            "
                pub fn add_to_linker_instance<T, D>(
                    inst: &mut {wt}::component::LinkerInstance<'_, T>,
                    {id_param}
                    {options_param}
                    host_getter: fn(&mut T) -> D::Data<'_>,
                ) -> {wt}::Result<()>
                    where
                        D: HostWithStore<T>,
                        for<'a> D::Data<'a>: {sync_bounds},
                        T: 'static {opt_t_send_bound},
                {{
            "
        );

        let gate = FeatureGate::open(&mut self.src, &iface.stability);
        for (ty, _name) in get_resources(self.resolve, id) {
            self.generator.generate_add_resource_to_linker(
                self.current_interface.map(|p| p.1),
                Some(&mut self.src),
                "inst",
                self.resolve,
                ty,
            );
        }

        for (_, func) in iface.functions.iter() {
            self.generate_add_function_to_linker(owner, func, "inst");
        }
        gate.close(&mut self.src);
        uwriteln!(self.src, "Ok(())");
        uwriteln!(self.src, "}}");

        match &self.named_import_id {
            Some(id_ty) => {
                let (id, _, _) = self.current_interface.unwrap();
                let wit_name = self.resolve.id_of(id).unwrap();
                uwriteln!(
                    self.src,
                    "
pub fn add_to_linker<T, D>(
    linker: &mut {wt}::component::Linker<T>,
    component: &{wt}::component::Component,
    mut lookup: impl FnMut(&str) -> {wt}::Result<{id_ty}>,
    {options_param}
    host_getter: fn(&mut T) -> D::Data<'_>,
) -> {wt}::Result<()>
    where
        D: HostWithStore<T>,
        for<'a> D::Data<'a>: {sync_bounds},
        T: 'static {opt_t_send_bound},
{{
    // Collect matching imports up front: iterating `imports`
    // borrows the engine (via `linker`) immutably while
    // `linker.instance(..)` needs a mutable borrow.
    let engine = linker.engine().clone();
    let component_ty = component.component_type();
    let mut matched = {wt}::component::__internal::Vec::new();
    for (name, item) in component_ty.imports(&engine) {{
        if item.is_implements({wit_name:?}) {{
            matched.push((name, lookup(name)?));
        }}
    }}
    for (name, id) in matched {{
        let mut inst = linker.instance(name)?;
        add_to_linker_instance::<T, D>(
            &mut inst,
            id,
            {options_param_forward}
            host_getter,
        )?;
    }}
    Ok(())
}}
            "
                );
            }
            None => {
                uwriteln!(
                    self.src,
                    "
pub fn add_to_linker<T, D>(
    linker: &mut {wt}::component::Linker<T>,
    {options_param}
    host_getter: fn(&mut T) -> D::Data<'_>,
) -> {wt}::Result<()>
    where
        D: HostWithStore<T>,
        for<'a> D::Data<'a>: {sync_bounds},
        T: 'static {opt_t_send_bound},
{{
    let mut inst = linker.instance(\"{name}\")?;
    add_to_linker_instance::<T, D>(&mut inst, {options_param_forward} host_getter)
}}
            "
                );
            }
        }
    }

    fn import_resource_drop_flags(&mut self, name: &str) -> FunctionFlags {
        self.generator.opts.imports.resource_drop_flags(
            self.resolve,
            self.current_interface.map(|p| p.1),
            name,
        )
    }

    fn generate_add_function_to_linker(&mut self, owner: TypeOwner, func: &Function, linker: &str) {
        let flags = self.generator.opts.imports.flags(
            self.resolve,
            self.current_interface.map(|p| p.1),
            func,
        );
        self.all_func_flags |= flags;
        let gate = FeatureGate::open(&mut self.src, &func.stability);
        uwrite!(
            self.src,
            "{linker}.{}(\"{}\", ",
            if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
                "func_wrap_concurrent"
            } else if flags.contains(FunctionFlags::ASYNC) {
                "func_wrap_async"
            } else {
                "func_wrap"
            },
            func.name
        );
        if self.named_import_id.is_some() {
            self.src.push_str("{ let id = id.clone(); ");
        }
        self.generate_guest_import_closure(owner, func, flags);
        if self.named_import_id.is_some() {
            self.src.push_str("}\n");
        }
        uwriteln!(self.src, ")?;");
        gate.close(&mut self.src);
    }

    fn generate_guest_import_closure(
        &mut self,
        owner: TypeOwner,
        func: &Function,
        flags: FunctionFlags,
    ) {
        // Generate the closure that's passed to a `Linker`, the final piece of
        // codegen here.

        let wt = self.generator.wasmtime_path();
        if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
            uwrite!(self.src, "move |caller: &{wt}::component::Accessor::<T>, (");
        } else {
            uwrite!(
                self.src,
                "move |mut caller: {wt}::StoreContextMut<'_, T>, ("
            );
        }
        for (i, _param) in func.params.iter().enumerate() {
            uwrite!(self.src, "arg{},", i);
        }
        self.src.push_str(") : (");

        for param in func.params.iter() {
            // Lift is required to be implied for this type, so we can't use
            // a borrowed type:
            self.print_ty(&param.ty, TypeMode::Owned);
            self.src.push_str(", ");
        }
        self.src.push_str(")| {\n");

        if flags.contains(FunctionFlags::TRACING) {
            if flags.contains(FunctionFlags::ASYNC) {
                self.src.push_str("use tracing::Instrument;\n");
            }

            uwrite!(
                self.src,
                "
                   let span = tracing::span!(
                       tracing::Level::TRACE,
                       \"wit-bindgen import\",
                       module = \"{}\",
                       function = \"{}\",
                   );
               ",
                match owner {
                    TypeOwner::Interface(id) => self.resolve.interfaces[id]
                        .name
                        .as_deref()
                        .unwrap_or("<no module>"),
                    TypeOwner::World(id) => &self.resolve.worlds[id].name,
                    TypeOwner::None => "<no owner>",
                },
                func.name,
            );
        }
        if self.named_import_id.is_some() {
            self.src.push_str("let id = id.clone(); ");
        }

        if flags.contains(FunctionFlags::ASYNC) {
            let ctor = if flags.contains(FunctionFlags::STORE) {
                "pin"
            } else {
                "new"
            };
            uwriteln!(
                self.src,
                "{wt}::component::__internal::Box::{ctor}(async move {{"
            );
        } else {
            // Only directly enter the span if the function is sync. Otherwise
            // we use tracing::Instrument to ensure that the span is not entered
            // across an await point.
            if flags.contains(FunctionFlags::TRACING) {
                self.push_str("let _enter = span.enter();\n");
            }
        }

        if flags.contains(FunctionFlags::TRACING) {
            let mut event_fields = func
                .params
                .iter()
                .enumerate()
                .map(|(i, param)| {
                    let name = to_rust_ident(&param.name);
                    formatting_for_arg(&name, i, param.ty, &self.resolve, flags)
                })
                .collect::<Vec<String>>();
            event_fields.push(format!("\"call\""));
            uwrite!(
                self.src,
                "tracing::event!(tracing::Level::TRACE, {});\n",
                event_fields.join(", ")
            );
        }

        if flags.contains(FunctionFlags::STORE) {
            if flags.contains(FunctionFlags::ASYNC) {
                uwriteln!(self.src, "let host = &caller.with_getter(host_getter);");
            } else {
                uwriteln!(
                    self.src,
                    "let access_cx = {wt}::AsContextMut::as_context_mut(&mut caller);"
                );
                uwriteln!(
                    self.src,
                    "let host = {wt}::component::Access::new(access_cx, host_getter);"
                );
            }
        } else {
            self.src
                .push_str("let host = &mut host_getter(caller.data_mut());\n");
        }
        let func_name = rust_function_name(func);
        let host_trait = match func.kind.resource() {
            None => match owner {
                TypeOwner::World(id) => format!(
                    "{}Imports",
                    rust::to_rust_upper_camel_case(&self.resolve.worlds[id].name)
                ),
                _ => "Host".to_string(),
            },
            Some(id) => {
                let resource = self.resolve.types[id]
                    .name
                    .as_ref()
                    .unwrap()
                    .to_upper_camel_case();
                format!("Host{resource}")
            }
        };

        if flags.contains(FunctionFlags::STORE) {
            uwrite!(
                self.src,
                "let r = <D as {host_trait}WithStore<T>>::{func_name}(host, "
            );
        } else {
            uwrite!(self.src, "let r = {host_trait}::{func_name}(host, ");
        }

        if self.named_import_id.is_some() {
            self.src.push_str("id, ");
        }

        for (i, _) in func.params.iter().enumerate() {
            uwrite!(self.src, "arg{},", i);
        }

        self.src.push_str(if flags.contains(FunctionFlags::ASYNC) {
            ").await;\n"
        } else {
            ");\n"
        });

        if flags.contains(FunctionFlags::TRACING) {
            uwrite!(
                self.src,
                "tracing::event!(tracing::Level::TRACE, {}, \"return\");",
                formatting_for_results(func.result, &self.resolve, flags)
            );
        }

        if !flags.contains(FunctionFlags::TRAPPABLE) {
            if func.result.is_some() {
                uwrite!(self.src, "Ok((r,))\n");
            } else {
                uwrite!(self.src, "Ok(r)\n");
            }
        } else if let Some((_, err, _)) = self.special_case_trappable_error(func) {
            let err = &self.resolve.types[resolve_type_definition_id(self.resolve, err)];
            let err_name = err.name.as_ref().unwrap();
            let owner = match err.owner {
                TypeOwner::Interface(i) => i,
                _ => unimplemented!(),
            };
            let convert_trait = match self.path_to_interface(owner) {
                Some(path) => format!("{path}::Host"),
                None => format!("Host"),
            };
            let convert = format!("{}::convert_{}", convert_trait, err_name.to_snake_case());
            let convert = if flags.contains(FunctionFlags::STORE) {
                if flags.contains(FunctionFlags::ASYNC) {
                    format!("caller.with(|mut host| {convert}(&mut host_getter(host.get()), e))")
                } else {
                    format!("{convert}(&mut host_getter(caller.data_mut()), e)")
                }
            } else {
                format!("{convert}(host, e)")
            };
            uwrite!(
                self.src,
                "Ok((match r {{
                    Ok(a) => Ok(a),
                    Err(e) => Err({wt}::ToWasmtimeResult::to_wasmtime_result({convert})?),
                }},))"
            );
        } else if func.result.is_some() {
            uwrite!(
                self.src,
                "Ok(({wt}::ToWasmtimeResult::to_wasmtime_result(r)?,))\n"
            );
        } else {
            uwrite!(self.src, "{wt}::ToWasmtimeResult::to_wasmtime_result(r)\n");
        }

        if flags.contains(FunctionFlags::ASYNC) {
            if flags.contains(FunctionFlags::TRACING) {
                self.src.push_str("}.instrument(span))\n");
            } else {
                self.src.push_str("})\n");
            }
        }

        self.src.push_str("}\n");
    }

    fn generate_function_trait_sig(&mut self, func: &Function, flags: FunctionFlags) {
        let wt = self.generator.wasmtime_path();
        self.rustdoc(&func.docs);

        self.push_str("fn ");
        self.push_str(&rust_function_name(func));
        if flags.contains(FunctionFlags::STORE | FunctionFlags::ASYNC) {
            uwrite!(self.src, "(accessor: &{wt}::component::Accessor<T, Self>, ");
        } else if flags.contains(FunctionFlags::STORE) {
            uwrite!(self.src, "(host: {wt}::component::Access<T, Self>, ");
        } else {
            self.push_str("(&mut self, ");
        }
        if let Some(id) = &self.named_import_id {
            uwrite!(self.src, "id: {id}, ");
        }
        self.generate_function_params(func);
        self.push_str(")");
        self.push_str(" -> ");

        if flags.contains(FunctionFlags::ASYNC) {
            uwrite!(self.src, "impl ::core::future::Future<Output = ");
        }

        self.all_func_flags |= flags;
        self.generate_function_result(func, flags);

        if flags.contains(FunctionFlags::ASYNC) {
            self.push_str("> + Send");
        }
    }

    fn generate_function_params(&mut self, func: &Function) {
        for param in func.params.iter() {
            let name = to_rust_ident(&param.name);
            self.push_str(&name);
            self.push_str(": ");
            self.print_ty(&param.ty, TypeMode::Owned);
            self.push_str(",");
        }
    }

    fn push_wasmtime_or_anyhow_result(&mut self) {
        let wt = self.generator.wasmtime_path();
        uwrite!(self.src, "{wt}::");
        if self.generator.opts.anyhow {
            self.push_str("anyhow::");
        }
        self.push_str("Result");
    }

    fn generate_function_result(&mut self, func: &Function, flags: FunctionFlags) {
        if !flags.contains(FunctionFlags::TRAPPABLE) {
            self.print_result_ty(func.result, TypeMode::Owned);
        } else if let Some((r, _id, error_typename)) = self.special_case_trappable_error(func) {
            // Functions which have a single result `result<ok,err>` get special
            // cased to use the host_wasmtime_rust::Error<err>, making it possible
            // for them to trap or use `?` to propagate their errors
            self.push_str("Result<");
            if let Some(ok) = r.ok {
                self.print_ty(&ok, TypeMode::Owned);
            } else {
                self.push_str("()");
            }
            self.push_str(",");
            self.push_str(&error_typename);
            self.push_str(">");
        } else {
            // All other functions get their return values wrapped in an wasmtime::Result.
            // Returning the anyhow::Error case can be used to trap.
            self.push_wasmtime_or_anyhow_result();
            self.push_str("<");
            self.print_result_ty(func.result, TypeMode::Owned);
            self.push_str(">");
        }
    }

    fn extract_typed_function(&mut self, func: &Function) -> (String, String) {
        let snake = func_field_name(self.resolve, func);
        let sig = self.typedfunc_sig(func, TypeMode::AllBorrowed("'_"));
        let extract =
            format!("*_instance.get_typed_func::<{sig}>(&mut store, &self.{snake})?.func()");
        (snake, extract)
    }

    fn define_rust_guest_export(
        &mut self,
        resolve: &Resolve,
        ns: Option<&WorldKey>,
        func: &Function,
    ) {
        let flags = self.generator.opts.exports.flags(resolve, ns, func);
        let (async_, async__, await_) = if flags.contains(FunctionFlags::ASYNC) {
            ("async", "_async", ".await")
        } else {
            ("", "", "")
        };
        let param_mode = if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
            TypeMode::Owned
        } else {
            TypeMode::AllBorrowed("'_")
        };
        let wt = self.generator.wasmtime_path();

        // First generate an accessor to get the raw `TypedFunc` itself.
        uwrite!(
            self.src,
            "pub fn func_{}(&self) -> {wt}::component::TypedFunc<{}> {{\n",
            func.item_name().to_snake_case(),
            self.typedfunc_sig(func, param_mode)
        );

        self.src.push_str("unsafe {\n");
        uwrite!(
            self.src,
            "{wt}::component::TypedFunc::<{}>",
            self.typedfunc_sig(func, param_mode)
        );
        let projection_to_func = if func.kind.resource().is_some() {
            ".funcs"
        } else {
            ""
        };
        uwriteln!(
            self.src,
            "::new_unchecked(self{projection_to_func}.{})",
            func_field_name(self.resolve, func),
        );
        self.src.push_str("}\n");
        self.src.push_str("}\n");

        // Next generate the actual function itself.
        self.rustdoc(&func.docs);
        uwrite!(
            self.src,
            "pub {async_} fn call_{}",
            func.item_name().to_snake_case(),
        );
        if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
            uwrite!(
                self.src,
                "<_T, _D>(&self, accessor: &{wt}::component::Accessor<_T, _D>, ",
            );
        } else {
            uwrite!(self.src, "<S: {wt}::AsContextMut>(&self, mut store: S, ",);
        }

        for (i, param) in func.params.iter().enumerate() {
            uwrite!(self.src, "arg{}: ", i);
            self.print_ty(&param.ty, param_mode);
            self.push_str(",");
        }

        uwrite!(self.src, ") -> {wt}::Result<");
        self.print_result_ty(func.result, TypeMode::Owned);
        uwrite!(self.src, ">");

        if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
            uwrite!(self.src, " where _T: Send, _D: {wt}::component::HasData");
        } else if flags.contains(FunctionFlags::ASYNC) {
            uwrite!(self.src, " where <S as {wt}::AsContext>::Data: Send");
        }
        uwrite!(self.src, "{{\n");

        if flags.contains(FunctionFlags::TRACING) {
            if flags.contains(FunctionFlags::ASYNC) {
                self.src.push_str("use tracing::Instrument;\n");
            }

            let ns = match ns {
                Some(key) => resolve.name_world_key(key),
                None => "default".to_string(),
            };
            self.src.push_str(&format!(
                "
                   let span = tracing::span!(
                       tracing::Level::TRACE,
                       \"wit-bindgen export\",
                       module = \"{ns}\",
                       function = \"{}\",
                   );
               ",
                func.name,
            ));

            if !flags.contains(FunctionFlags::ASYNC) {
                self.src.push_str(
                    "
                   let _enter = span.enter();
                   ",
                );
            }
        }

        uwriteln!(
            self.src,
            "let callee = self.func_{}();",
            func.item_name().to_snake_case(),
        );

        self.src.push_str("let (");
        if func.result.is_some() {
            uwrite!(self.src, "ret0,");
        }

        if flags.contains(FunctionFlags::ASYNC | FunctionFlags::STORE) {
            uwrite!(self.src, ") = callee.call_concurrent(accessor, (");
        } else {
            uwrite!(
                self.src,
                ") = callee.call{async__}(store.as_context_mut(), ("
            );
        };

        for (i, _) in func.params.iter().enumerate() {
            uwrite!(self.src, "arg{}, ", i);
        }

        let instrument = if flags.contains(FunctionFlags::ASYNC | FunctionFlags::TRACING) {
            ".instrument(span.clone())"
        } else {
            ""
        };
        uwriteln!(self.src, ")){instrument}{await_}?;");

        self.src.push_str("Ok(");
        if func.result.is_some() {
            self.src.push_str("ret0");
        } else {
            self.src.push_str("()");
        }
        self.src.push_str(")\n");

        // End function body
        self.src.push_str("}\n");
    }

    fn rustdoc(&mut self, docs: &Docs) {
        let docs = match &docs.contents {
            Some(docs) => docs,
            None => return,
        };
        for line in docs.trim().lines() {
            self.push_str("/// ");
            self.push_str(line);
            self.push_str("\n");
        }
    }

    fn path_to_root(&self) -> String {
        let mut path_to_root = String::new();
        if let Some((_, key, kind)) = self.current_interface {
            match kind {
                InterfaceKind::Export | InterfaceKind::Named => path_to_root.push_str("super::"),
                InterfaceKind::Import => {}
            }
            match key {
                WorldKey::Name(_) => {
                    path_to_root.push_str("super::");
                }
                WorldKey::Interface(_) => {
                    path_to_root.push_str("super::super::super::");
                }
            }
        }
        path_to_root
    }

    fn partition_concurrent_funcs<'b>(
        &mut self,
        funcs: impl IntoIterator<Item = &'b Function>,
    ) -> FunctionPartitioning<'b> {
        let key = self.current_interface.map(|p| p.1);
        let (with_store, without_store) = funcs
            .into_iter()
            .map(|func| {
                let flags = self.generator.opts.imports.flags(self.resolve, key, func);
                (func, flags)
            })
            .partition(|(_, flags)| flags.contains(FunctionFlags::STORE));
        FunctionPartitioning {
            with_store,
            without_store,
        }
    }

    fn generate_trait(
        &mut self,
        trait_name: &str,
        functions: &[&Function],
        extra_functions: &[ExtraTraitMethod<'_>],
        resources: &[(TypeId, &str)],
    ) -> GeneratedTrait {
        let mut ret = GeneratedTrait::default();
        let wt = self.generator.wasmtime_path();
        let partition = self.partition_concurrent_funcs(functions.iter().copied());

        for (_, flags) in partition.with_store.iter().chain(&partition.without_store) {
            ret.all_func_flags |= *flags;
        }

        let mut with_store_supertraits = vec![format!("{wt}::component::HasData")];
        let mut without_store_supertraits = vec![];
        for (id, name) in resources {
            let camel = name.to_upper_camel_case();
            without_store_supertraits.push(format!("Host{camel}"));
            let funcs = self.partition_concurrent_funcs(get_resource_functions(self.resolve, *id));
            for (_, flags) in funcs.with_store.iter().chain(&funcs.without_store) {
                ret.all_func_flags |= *flags;
            }
            ret.all_func_flags |= self.import_resource_drop_flags(name);
            with_store_supertraits.push(format!("Host{camel}WithStore<T>"));
        }
        if ret.all_func_flags.contains(FunctionFlags::ASYNC) {
            with_store_supertraits.push("Send".to_string());
            without_store_supertraits.push("Send".to_string());
        }

        uwriteln!(
            self.src,
            "pub trait {trait_name}WithStore<T>: {} {{",
            with_store_supertraits.join(" + "),
        );
        ret.with_store_name = Some(format!("{trait_name}WithStore<T>"));

        let mut extra_with_store_function = false;
        for extra in extra_functions {
            match extra {
                ExtraTraitMethod::ResourceDrop { name } => {
                    let flags = self.import_resource_drop_flags(name);
                    if !flags.contains(FunctionFlags::STORE) {
                        continue;
                    }
                    let camel = name.to_upper_camel_case();

                    if flags.contains(FunctionFlags::ASYNC) {
                        uwrite!(
                            self.src,
                            "
fn drop(accessor: &{wt}::component::Accessor<T, Self>, rep: {wt}::component::Resource<{camel}>)
    -> impl ::core::future::Future<Output =
"
                        );
                        self.push_wasmtime_or_anyhow_result();
                        self.push_str("<()>> + Send where Self: Sized;");
                    } else {
                        uwrite!(
                            self.src,
                            "
fn drop(accessor: {wt}::component::Access<T, Self>, rep: {wt}::component::Resource<{camel}>)
    ->
"
                        );
                        self.push_wasmtime_or_anyhow_result();
                        self.push_str("<()>;");
                    }

                    extra_with_store_function = true;
                }
                ExtraTraitMethod::ErrorConvert { .. } => {}
            }
        }

        for (func, flags) in partition.with_store.iter() {
            self.generate_function_trait_sig(func, *flags);
            self.push_str(";\n");
        }
        uwriteln!(self.src, "}}");

        // If `*WithStore` is empty, generate a blanket impl for the trait since
        // it's otherwise not necessary to implement it manually.
        if partition.with_store.is_empty() && !extra_with_store_function {
            uwriteln!(
                self.src,
                "impl<H: ?Sized, T> {trait_name}WithStore<T> for H"
            );
            uwriteln!(self.src, " where H: {}", with_store_supertraits.join(" + "));

            uwriteln!(self.src, "{{}}");
        }

        uwriteln!(
            self.src,
            "pub trait {trait_name}: {} {{",
            without_store_supertraits.join(" + ")
        );
        ret.name = trait_name.to_string();
        for (func, flags) in partition.without_store.iter() {
            self.generate_function_trait_sig(func, *flags);
            self.push_str(";\n");
        }

        for extra in extra_functions {
            match extra {
                ExtraTraitMethod::ResourceDrop { name } => {
                    let flags = self.import_resource_drop_flags(name);
                    ret.all_func_flags |= flags;
                    if flags.contains(FunctionFlags::STORE) {
                        continue;
                    }
                    let camel = name.to_upper_camel_case();
                    uwrite!(
                        self.src,
                        "fn drop(&mut self, rep: {wt}::component::Resource<{camel}>) -> "
                    );
                    if flags.contains(FunctionFlags::ASYNC) {
                        uwrite!(self.src, "impl ::core::future::Future<Output =");
                    }
                    self.push_wasmtime_or_anyhow_result();
                    self.push_str("<()>");
                    if flags.contains(FunctionFlags::ASYNC) {
                        uwrite!(self.src, "> + Send");
                    }
                    uwrite!(self.src, ";");
                }
                ExtraTraitMethod::ErrorConvert { name, id } => {
                    let root = self.path_to_root();
                    let custom_name = &self.generator.trappable_errors[id];
                    let snake = name.to_snake_case();
                    let camel = name.to_upper_camel_case();
                    uwrite!(
                        self.src,
                        "
fn convert_{snake}(&mut self, err: {root}{custom_name}) ->
                        "
                    );
                    self.push_wasmtime_or_anyhow_result();
                    uwrite!(self.src, "<{camel}>;");
                }
            }
        }

        uwriteln!(self.src, "}}");

        if self.generator.opts.skip_mut_forwarding_impls {
            return ret;
        }

        // Generate impl HostResource for &mut HostResource
        let maybe_send = if ret.all_func_flags.contains(FunctionFlags::ASYNC) {
            "+ Send"
        } else {
            ""
        };
        uwriteln!(
            self.src,
            "impl <_T: {trait_name} + ?Sized {maybe_send}> {trait_name} for &mut _T {{"
        );
        for (func, flags) in partition.without_store.iter() {
            self.generate_function_trait_sig(func, *flags);
            uwriteln!(self.src, "{{");
            if flags.contains(FunctionFlags::ASYNC) {
                uwriteln!(self.src, "async move {{");
            }
            uwrite!(
                self.src,
                "{trait_name}::{}(*self,",
                rust_function_name(func)
            );
            if self.named_import_id.is_some() {
                self.src.push_str("id,");
            }
            for param in func.params.iter() {
                uwrite!(self.src, "{},", to_rust_ident(&param.name));
            }
            uwrite!(self.src, ")");
            if flags.contains(FunctionFlags::ASYNC) {
                uwrite!(self.src, ".await\n}}");
            }
            uwriteln!(self.src, "}}");
        }
        for extra in extra_functions {
            match extra {
                ExtraTraitMethod::ResourceDrop { name } => {
                    let flags = self.import_resource_drop_flags(name);
                    if flags.contains(FunctionFlags::STORE) {
                        continue;
                    }
                    let camel = name.to_upper_camel_case();
                    let mut await_ = "";
                    if flags.contains(FunctionFlags::ASYNC) {
                        self.src.push_str("async ");
                        await_ = ".await";
                    }
                    uwrite!(
                        self.src,
                        "fn drop(&mut self, rep: {wt}::component::Resource<{camel}>) -> ",
                    );
                    self.push_wasmtime_or_anyhow_result();
                    uwriteln!(
                        self.src,
                        "<()> {{
    {trait_name}::drop(*self, rep){await_}
}}
                        ",
                    );
                }
                ExtraTraitMethod::ErrorConvert { name, id } => {
                    let root = self.path_to_root();
                    let custom_name = &self.generator.trappable_errors[id];
                    let snake = name.to_snake_case();
                    let camel = name.to_upper_camel_case();
                    uwrite!(
                        self.src,
                        "fn convert_{snake}(&mut self, err: {root}{custom_name}) -> ",
                    );
                    self.push_wasmtime_or_anyhow_result();
                    uwriteln!(
                        self.src,
                        "<{camel}> {{
    {trait_name}::convert_{snake}(*self, err)
}}
                        ",
                    );
                }
            }
        }
        uwriteln!(self.src, "}}");

        ret
    }
}

enum ExtraTraitMethod<'a> {
    ResourceDrop { name: &'a str },
    ErrorConvert { name: &'a str, id: TypeId },
}

struct FunctionPartitioning<'a> {
    without_store: Vec<(&'a Function, FunctionFlags)>,
    with_store: Vec<(&'a Function, FunctionFlags)>,
}

#[derive(Default)]
struct GeneratedTrait {
    name: String,
    with_store_name: Option<String>,
    all_func_flags: FunctionFlags,
}

impl<'a> RustGenerator<'a> for InterfaceGenerator<'a> {
    fn resolve(&self) -> &'a Resolve {
        self.resolve
    }

    fn ownership(&self) -> Ownership {
        self.generator.opts.ownership
    }

    fn path_to_interface(&self, interface: InterfaceId) -> Option<String> {
        if let Some((cur, _, kind)) = self.current_interface {
            // If `interface` is `cur`, then we're in the same module and need
            // to path to the interface. If we're generating for a named import,
            // however, that's not true since the types live elsewhere, so skip
            // that case.
            if cur == interface && kind != InterfaceKind::Named {
                return None;
            }
        }
        let mut path_to_root = self.path_to_root();
        match &self.generator.interface_names[&interface] {
            InterfaceName::Remapped { name_at_root, .. } => path_to_root.push_str(name_at_root),
            InterfaceName::Path(path) => {
                for (i, name) in path.iter().enumerate() {
                    if i > 0 {
                        path_to_root.push_str("::");
                    }
                    path_to_root.push_str(name);
                }
            }
        }
        Some(path_to_root)
    }

    fn push_str(&mut self, s: &str) {
        self.src.push_str(s);
    }

    fn info(&self, ty: TypeId) -> TypeInfo {
        self.generator.types.get(ty)
    }

    fn is_imported_interface(&self, interface: InterfaceId) -> bool {
        if let Some((cur, _, kind)) = self.current_interface {
            if cur == interface {
                return kind != InterfaceKind::Export;
            }
        }
        self.generator.import_interfaces.contains_key(&interface)
    }

    fn wasmtime_path(&self) -> String {
        self.generator.wasmtime_path()
    }
}

#[derive(Default)]
struct LinkOptionsBuilder {
    unstable_features: BTreeSet<String>,
}
impl LinkOptionsBuilder {
    fn has_any(&self) -> bool {
        !self.unstable_features.is_empty()
    }
    fn add_world(&mut self, resolve: &Resolve, id: &WorldId) {
        let world = &resolve.worlds[*id];

        self.add_stability(&world.stability);

        for (_, import) in world.imports.iter() {
            match import {
                WorldItem::Interface { id, stability, .. } => {
                    self.add_stability(stability);
                    self.add_interface(resolve, id);
                }
                WorldItem::Function(f) => {
                    self.add_stability(&f.stability);
                }
                WorldItem::Type { id, .. } => {
                    self.add_type(resolve, id);
                }
            }
        }
    }
    fn add_interface(&mut self, resolve: &Resolve, id: &InterfaceId) {
        let interface = &resolve.interfaces[*id];

        self.add_stability(&interface.stability);

        for (_, t) in interface.types.iter() {
            self.add_type(resolve, t);
        }
        for (_, f) in interface.functions.iter() {
            self.add_stability(&f.stability);
        }
    }
    fn add_type(&mut self, resolve: &Resolve, id: &TypeId) {
        let t = &resolve.types[*id];
        self.add_stability(&t.stability);
    }
    fn add_stability(&mut self, stability: &Stability) {
        match stability {
            Stability::Unstable { feature, .. } => {
                self.unstable_features.insert(feature.clone());
            }
            Stability::Stable { .. } | Stability::Unknown => {}
        }
    }
    fn write_struct(&self, src: &mut Source) {
        if !self.has_any() {
            return;
        }

        let mut unstable_features = self.unstable_features.iter().cloned().collect::<Vec<_>>();
        unstable_features.sort();

        uwriteln!(
            src,
            "
            /// Link-time configurations.
            #[derive(Clone, Debug, Default)]
            pub struct LinkOptions {{
            "
        );

        for feature in unstable_features.iter() {
            let feature_rust_name = feature.to_snake_case();
            uwriteln!(src, "{feature_rust_name}: bool,");
        }

        uwriteln!(src, "}}");
        uwriteln!(src, "impl LinkOptions {{");

        for feature in unstable_features.iter() {
            let feature_rust_name = feature.to_snake_case();
            uwriteln!(
                src,
                "
                /// Enable members marked as `@unstable(feature = {feature})`
                pub fn {feature_rust_name}(&mut self, enabled: bool) -> &mut Self {{
                    self.{feature_rust_name} = enabled;
                    self
                }}
            "
            );
        }

        uwriteln!(src, "}}");
    }
    fn write_impl_from_world(&self, src: &mut Source, path: &str) {
        if !self.has_any() {
            return;
        }

        let mut unstable_features = self.unstable_features.iter().cloned().collect::<Vec<_>>();
        unstable_features.sort();

        uwriteln!(
            src,
            "
            impl core::convert::From<LinkOptions> for {path}::LinkOptions {{
                fn from(src: LinkOptions) -> Self {{
                    (&src).into()
                }}
            }}

            impl core::convert::From<&LinkOptions> for {path}::LinkOptions {{
                fn from(src: &LinkOptions) -> Self {{
                    let mut dest = Self::default();
        "
        );

        for feature in unstable_features.iter() {
            let feature_rust_name = feature.to_snake_case();
            uwriteln!(src, "dest.{feature_rust_name}(src.{feature_rust_name});");
        }

        uwriteln!(
            src,
            "
                    dest
                }}
            }}
        "
        );
    }
}

struct FeatureGate {
    close: bool,
}
impl FeatureGate {
    fn open(src: &mut Source, stability: &Stability) -> FeatureGate {
        let close = if let Stability::Unstable { feature, .. } = stability {
            let feature_rust_name = feature.to_snake_case();
            uwrite!(src, "if options.{feature_rust_name} {{");
            true
        } else {
            false
        };
        Self { close }
    }

    fn close(self, src: &mut Source) {
        if self.close {
            uwriteln!(src, "}}");
        }
    }
}

/// Produce a string for tracing a function argument.
fn formatting_for_arg(
    name: &str,
    index: usize,
    ty: Type,
    resolve: &Resolve,
    flags: FunctionFlags,
) -> String {
    if !flags.contains(FunctionFlags::VERBOSE_TRACING) && type_contains_lists(ty, resolve) {
        return format!("{name} = tracing::field::debug(\"...\")");
    }

    // Normal tracing.
    format!("{name} = tracing::field::debug(&arg{index})")
}

/// Produce a string for tracing function results.
fn formatting_for_results(result: Option<Type>, resolve: &Resolve, flags: FunctionFlags) -> String {
    let contains_lists = match result {
        Some(ty) => type_contains_lists(ty, resolve),
        None => false,
    };

    if !flags.contains(FunctionFlags::VERBOSE_TRACING) && contains_lists {
        return format!("result = tracing::field::debug(\"...\")");
    }

    // Normal tracing.
    format!("result = tracing::field::debug(&r)")
}

/// Test whether the given type contains lists.
///
/// Here, a `string` is not considered a list.
fn type_contains_lists(ty: Type, resolve: &Resolve) -> bool {
    match ty {
        Type::Id(id) => match &resolve.types[id].kind {
            TypeDefKind::Resource
            | TypeDefKind::Unknown
            | TypeDefKind::Flags(_)
            | TypeDefKind::Handle(_)
            | TypeDefKind::Enum(_)
            | TypeDefKind::Stream(_)
            | TypeDefKind::Future(_) => false,
            TypeDefKind::Option(ty) => type_contains_lists(*ty, resolve),
            TypeDefKind::Result(Result_ { ok, err }) => {
                option_type_contains_lists(*ok, resolve)
                    || option_type_contains_lists(*err, resolve)
            }
            TypeDefKind::Record(record) => record
                .fields
                .iter()
                .any(|field| type_contains_lists(field.ty, resolve)),
            TypeDefKind::Tuple(tuple) => tuple
                .types
                .iter()
                .any(|ty| type_contains_lists(*ty, resolve)),
            TypeDefKind::Variant(variant) => variant
                .cases
                .iter()
                .any(|case| option_type_contains_lists(case.ty, resolve)),
            TypeDefKind::Type(ty) => type_contains_lists(*ty, resolve),
            TypeDefKind::List(_) => true,
            TypeDefKind::Map(k, v) => {
                type_contains_lists(*k, resolve) || type_contains_lists(*v, resolve)
            }
            TypeDefKind::FixedLengthList(..) => todo!(),
        },

        // Technically strings are lists too, but we ignore that here because
        // they're usually short.
        _ => false,
    }
}

fn option_type_contains_lists(ty: Option<Type>, resolve: &Resolve) -> bool {
    match ty {
        Some(ty) => type_contains_lists(ty, resolve),
        None => false,
    }
}

/// When an interface `use`s a type from another interface, it creates a new TypeId
/// referring to the definition TypeId. Chase this chain of references down to
/// a TypeId for type's definition.
fn resolve_type_definition_id(resolve: &Resolve, mut id: TypeId) -> TypeId {
    loop {
        match resolve.types[id].kind {
            TypeDefKind::Type(Type::Id(def_id)) => id = def_id,
            _ => return id,
        }
    }
}

fn rust_function_name(func: &Function) -> String {
    match func.kind {
        FunctionKind::Constructor(_) => "new".to_string(),
        FunctionKind::Method(_)
        | FunctionKind::Static(_)
        | FunctionKind::AsyncMethod(_)
        | FunctionKind::AsyncStatic(_)
        | FunctionKind::Freestanding
        | FunctionKind::AsyncFreestanding => to_rust_ident(func.item_name()),
    }
}

fn func_field_name(resolve: &Resolve, func: &Function) -> String {
    let mut name = String::new();
    match func.kind {
        FunctionKind::Method(id) | FunctionKind::AsyncMethod(id) => {
            name.push_str("method-");
            name.push_str(resolve.types[id].name.as_ref().unwrap());
            name.push_str("-");
        }
        FunctionKind::Static(id) | FunctionKind::AsyncStatic(id) => {
            name.push_str("static-");
            name.push_str(resolve.types[id].name.as_ref().unwrap());
            name.push_str("-");
        }
        FunctionKind::Constructor(id) => {
            name.push_str("constructor-");
            name.push_str(resolve.types[id].name.as_ref().unwrap());
            name.push_str("-");
        }
        FunctionKind::Freestanding | FunctionKind::AsyncFreestanding => {}
    }
    name.push_str(func.item_name());
    name.to_snake_case()
}

fn get_resources<'a>(
    resolve: &'a Resolve,
    id: InterfaceId,
) -> impl Iterator<Item = (TypeId, &'a str)> + 'a {
    resolve.interfaces[id]
        .types
        .iter()
        .filter_map(move |(name, ty)| match &resolve.types[*ty].kind {
            TypeDefKind::Resource => Some((*ty, name.as_str())),
            _ => None,
        })
}

fn get_resource_functions<'a>(resolve: &'a Resolve, resource_id: TypeId) -> Vec<&'a Function> {
    let resource = &resolve.types[resource_id];
    match resource.owner {
        TypeOwner::World(id) => resolve.worlds[id]
            .imports
            .values()
            .filter_map(|item| match item {
                WorldItem::Function(f) => Some(f),
                _ => None,
            })
            .filter(|f| f.kind.resource() == Some(resource_id))
            .collect(),
        TypeOwner::Interface(id) => resolve.interfaces[id]
            .functions
            .values()
            .filter(|f| f.kind.resource() == Some(resource_id))
            .collect::<Vec<_>>(),
        TypeOwner::None => {
            panic!("A resource must be owned by a world or interface");
        }
    }
}

fn get_world_resources<'a>(
    resolve: &'a Resolve,
    id: WorldId,
) -> impl Iterator<Item = (TypeId, &'a str)> + 'a {
    resolve.worlds[id]
        .imports
        .iter()
        .filter_map(move |(name, item)| match item {
            WorldItem::Type { id, .. } => match resolve.types[*id].kind {
                TypeDefKind::Resource => Some(match name {
                    WorldKey::Name(s) => (*id, s.as_str()),
                    WorldKey::Interface(_) => unreachable!(),
                }),
                _ => None,
            },
            _ => None,
        })
}
