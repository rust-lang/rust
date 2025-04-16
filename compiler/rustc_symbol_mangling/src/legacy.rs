use std::fmt::{self, Write};
use std::mem::{self, discriminant};

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_hashes::Hash64;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use rustc_middle::bug;
use rustc_middle::ty::print::{PrettyPrinter, Print, PrintError, Printer};
use rustc_middle::ty::{
    self, GenericArg, GenericArgKind, Instance, ReifyReason, Ty, TyCtxt, TypeVisitableExt,
};
use tracing::debug;

pub(super) fn mangle<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    instantiating_crate: Option<CrateNum>,
) -> String {
    let def_id = instance.def_id();

    // We want to compute the "type" of this item. Unfortunately, some
    // kinds of items (e.g., synthetic static allocations from const eval)
    // don't have a proper implementation for the `type_of` query. So walk
    // back up the find the closest parent that DOES have a type.
    let mut ty_def_id = def_id;
    let instance_ty;
    loop {
        let key = tcx.def_key(ty_def_id);
        match key.disambiguated_data.data {
            DefPathData::TypeNs(_)
            | DefPathData::ValueNs(_)
            | DefPathData::Closure
            | DefPathData::SyntheticCoroutineBody => {
                instance_ty = tcx.type_of(ty_def_id).instantiate_identity();
                debug!(?instance_ty);
                break;
            }
            _ => {
                // if we're making a symbol for something, there ought
                // to be a value or type-def or something in there
                // *somewhere*
                ty_def_id.index = key.parent.unwrap_or_else(|| {
                    bug!(
                        "finding type for {:?}, encountered def-id {:?} with no \
                         parent",
                        def_id,
                        ty_def_id
                    );
                });
            }
        }
    }

    // Erase regions because they may not be deterministic when hashed
    // and should not matter anyhow.
    let instance_ty = tcx.erase_regions(instance_ty);

    let hash = get_symbol_hash(tcx, instance, instance_ty, instantiating_crate);

    let mut printer = SymbolPrinter { tcx, path: SymbolPath::new(), keep_within_component: false };
    printer
        .print_def_path(
            def_id,
            if let ty::InstanceKind::DropGlue(_, _)
            | ty::InstanceKind::AsyncDropGlueCtorShim(_, _) = instance.def
            {
                // Add the name of the dropped type to the symbol name
                &*instance.args
            } else {
                &[]
            },
        )
        .unwrap();

    match instance.def {
        ty::InstanceKind::ThreadLocalShim(..) => {
            printer.write_str("{{tls-shim}}").unwrap();
        }
        ty::InstanceKind::VTableShim(..) => {
            printer.write_str("{{vtable-shim}}").unwrap();
        }
        ty::InstanceKind::ReifyShim(_, reason) => {
            printer.write_str("{{reify-shim").unwrap();
            match reason {
                Some(ReifyReason::FnPtr) => printer.write_str("-fnptr").unwrap(),
                Some(ReifyReason::Vtable) => printer.write_str("-vtable").unwrap(),
                None => (),
            }
            printer.write_str("}}").unwrap();
        }
        // FIXME(async_closures): This shouldn't be needed when we fix
        // `Instance::ty`/`Instance::def_id`.
        ty::InstanceKind::ConstructCoroutineInClosureShim { receiver_by_ref, .. } => {
            printer
                .write_str(if receiver_by_ref { "{{by-move-shim}}" } else { "{{by-ref-shim}}" })
                .unwrap();
        }
        _ => {}
    }

    printer.path.finish(hash)
}

fn get_symbol_hash<'tcx>(
    tcx: TyCtxt<'tcx>,

    // instance this name will be for
    instance: Instance<'tcx>,

    // type of the item, without any generic
    // parameters instantiated; this is
    // included in the hash as a kind of
    // safeguard.
    item_type: Ty<'tcx>,

    instantiating_crate: Option<CrateNum>,
) -> Hash64 {
    let def_id = instance.def_id();
    let args = instance.args;
    debug!("get_symbol_hash(def_id={:?}, parameters={:?})", def_id, args);

    tcx.with_stable_hashing_context(|mut hcx| {
        let mut hasher = StableHasher::new();

        // the main symbol name is not necessarily unique; hash in the
        // compiler's internal def-path, guaranteeing each symbol has a
        // truly unique path
        tcx.def_path_hash(def_id).hash_stable(&mut hcx, &mut hasher);

        // Include the main item-type. Note that, in this case, the
        // assertions about `has_param` may not hold, but this item-type
        // ought to be the same for every reference anyway.
        assert!(!item_type.has_erasable_regions());
        hcx.while_hashing_spans(false, |hcx| {
            item_type.hash_stable(hcx, &mut hasher);

            // If this is a function, we hash the signature as well.
            // This is not *strictly* needed, but it may help in some
            // situations, see the `run-make/a-b-a-linker-guard` test.
            if let ty::FnDef(..) = item_type.kind() {
                item_type.fn_sig(tcx).hash_stable(hcx, &mut hasher);
            }

            // also include any type parameters (for generic items)
            args.hash_stable(hcx, &mut hasher);

            if let Some(instantiating_crate) = instantiating_crate {
                tcx.def_path_hash(instantiating_crate.as_def_id())
                    .stable_crate_id()
                    .hash_stable(hcx, &mut hasher);
            }

            // We want to avoid accidental collision between different types of instances.
            // Especially, `VTableShim`s and `ReifyShim`s may overlap with their original
            // instances without this.
            discriminant(&instance.def).hash_stable(hcx, &mut hasher);
        });

        // 64 bits should be enough to avoid collisions.
        hasher.finish::<Hash64>()
    })
}

// Follow C++ namespace-mangling style, see
// https://en.wikipedia.org/wiki/Name_mangling for more info.
//
// It turns out that on macOS you can actually have arbitrary symbols in
// function names (at least when given to LLVM), but this is not possible
// when using unix's linker. Perhaps one day when we just use a linker from LLVM
// we won't need to do this name mangling. The problem with name mangling is
// that it seriously limits the available characters. For example we can't
// have things like &T in symbol names when one would theoretically
// want them for things like impls of traits on that type.
//
// To be able to work on all platforms and get *some* reasonable output, we
// use C++ name-mangling.
#[derive(Debug)]
struct SymbolPath {
    result: String,
    temp_buf: String,
}

impl SymbolPath {
    fn new() -> Self {
        let mut result =
            SymbolPath { result: String::with_capacity(64), temp_buf: String::with_capacity(16) };
        result.result.push_str("_ZN"); // _Z == Begin name-sequence, N == nested
        result
    }

    fn finalize_pending_component(&mut self) {
        if !self.temp_buf.is_empty() {
            let _ = write!(self.result, "{}{}", self.temp_buf.len(), self.temp_buf);
            self.temp_buf.clear();
        }
    }

    fn finish(mut self, hash: Hash64) -> String {
        self.finalize_pending_component();
        // E = end name-sequence
        let _ = write!(self.result, "17h{hash:016x}E");
        self.result
    }
}

struct SymbolPrinter<'tcx> {
    tcx: TyCtxt<'tcx>,
    path: SymbolPath,

    // When `true`, `finalize_pending_component` isn't used.
    // This is needed when recursing into `path_qualified`,
    // or `path_generic_args`, as any nested paths are
    // logically within one component.
    keep_within_component: bool,
}

// HACK(eddyb) this relies on using the `fmt` interface to get
// `PrettyPrinter` aka pretty printing of e.g. types in paths,
// symbol names should have their own printing machinery.

impl<'tcx> Printer<'tcx> for SymbolPrinter<'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn print_region(&mut self, _region: ty::Region<'_>) -> Result<(), PrintError> {
        Ok(())
    }

    fn print_type(&mut self, ty: Ty<'tcx>) -> Result<(), PrintError> {
        match *ty.kind() {
            // Print all nominal types as paths (unlike `pretty_print_type`).
            ty::FnDef(def_id, args)
            | ty::Alias(ty::Projection | ty::Opaque, ty::AliasTy { def_id, args, .. })
            | ty::Closure(def_id, args)
            | ty::CoroutineClosure(def_id, args)
            | ty::Coroutine(def_id, args) => self.print_def_path(def_id, args),

            // The `pretty_print_type` formatting of array size depends on
            // -Zverbose-internals flag, so we cannot reuse it here.
            ty::Array(ty, size) => {
                self.write_str("[")?;
                self.print_type(ty)?;
                self.write_str("; ")?;
                if let Some(size) = size.try_to_target_usize(self.tcx()) {
                    write!(self, "{size}")?
                } else if let ty::ConstKind::Param(param) = size.kind() {
                    param.print(self)?
                } else {
                    self.write_str("_")?
                }
                self.write_str("]")?;
                Ok(())
            }

            ty::Alias(ty::Inherent, _) => panic!("unexpected inherent projection"),

            _ => self.pretty_print_type(ty),
        }
    }

    fn print_dyn_existential(
        &mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<(), PrintError> {
        let mut first = true;
        for p in predicates {
            if !first {
                write!(self, "+")?;
            }
            first = false;
            p.print(self)?;
        }
        Ok(())
    }

    fn print_const(&mut self, ct: ty::Const<'tcx>) -> Result<(), PrintError> {
        // only print integers
        match ct.kind() {
            ty::ConstKind::Value(cv) if cv.ty.is_integral() => {
                // The `pretty_print_const` formatting depends on -Zverbose-internals
                // flag, so we cannot reuse it here.
                let scalar = cv.valtree.unwrap_leaf();
                let signed = matches!(cv.ty.kind(), ty::Int(_));
                write!(
                    self,
                    "{:#?}",
                    ty::ConstInt::new(scalar, signed, cv.ty.is_ptr_sized_integral())
                )?;
            }
            _ => self.write_str("_")?,
        }
        Ok(())
    }

    fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError> {
        self.write_str(self.tcx.crate_name(cnum).as_str())?;
        Ok(())
    }
    fn path_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        // Similar to `pretty_path_qualified`, but for the other
        // types that are printed as paths (see `print_type` above).
        match self_ty.kind() {
            ty::FnDef(..)
            | ty::Alias(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Coroutine(..)
                if trait_ref.is_none() =>
            {
                self.print_type(self_ty)
            }

            _ => self.pretty_path_qualified(self_ty, trait_ref),
        }
    }

    fn path_append_impl(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        _disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.pretty_path_append_impl(
            |cx| {
                print_prefix(cx)?;

                if cx.keep_within_component {
                    // HACK(eddyb) print the path similarly to how `FmtPrinter` prints it.
                    cx.write_str("::")?;
                } else {
                    cx.path.finalize_pending_component();
                }

                Ok(())
            },
            self_ty,
            trait_ref,
        )
    }
    fn path_append(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError> {
        print_prefix(self)?;

        // Skip `::{{extern}}` blocks and `::{{constructor}}` on tuple/unit structs.
        if let DefPathData::ForeignMod | DefPathData::Ctor = disambiguated_data.data {
            return Ok(());
        }

        if self.keep_within_component {
            // HACK(eddyb) print the path similarly to how `FmtPrinter` prints it.
            self.write_str("::")?;
        } else {
            self.path.finalize_pending_component();
        }

        write!(self, "{}", disambiguated_data.data)?;

        Ok(())
    }
    fn path_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        print_prefix(self)?;

        let args =
            args.iter().cloned().filter(|arg| !matches!(arg.unpack(), GenericArgKind::Lifetime(_)));

        if args.clone().next().is_some() {
            self.generic_delimiters(|cx| cx.comma_sep(args))
        } else {
            Ok(())
        }
    }

    fn print_impl_path(
        &mut self,
        impl_def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        let self_ty = self.tcx.type_of(impl_def_id);
        let impl_trait_ref = self.tcx.impl_trait_ref(impl_def_id);
        let generics = self.tcx.generics_of(impl_def_id);
        // We have two cases to worry about here:
        // 1. We're printing a nested item inside of an impl item, like an inner
        // function inside of a method. Due to the way that def path printing works,
        // we'll render this something like `<Ty as Trait>::method::inner_fn`
        // but we have no substs for this impl since it's not really inheriting
        // generics from the outer item. We need to use the identity substs, and
        // to normalize we need to use the correct param-env too.
        // 2. We're mangling an item with identity substs. This seems to only happen
        // when generating coverage, since we try to generate coverage for unused
        // items too, and if something isn't monomorphized then we necessarily don't
        // have anything to substitute the instance with.
        // NOTE: We don't support mangling partially substituted but still polymorphic
        // instances, like `impl<A> Tr<A> for ()` where `A` is substituted w/ `(T,)`.
        let (typing_env, mut self_ty, mut impl_trait_ref) = if generics.count() > args.len()
            || &args[..generics.count()]
                == self
                    .tcx
                    .erase_regions(ty::GenericArgs::identity_for_item(self.tcx, impl_def_id))
                    .as_slice()
        {
            (
                ty::TypingEnv::post_analysis(self.tcx, impl_def_id),
                self_ty.instantiate_identity(),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate_identity()),
            )
        } else {
            assert!(
                !args.has_non_region_param(),
                "should not be mangling partially substituted \
                polymorphic instance: {impl_def_id:?} {args:?}"
            );
            (
                ty::TypingEnv::fully_monomorphized(),
                self_ty.instantiate(self.tcx, args),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate(self.tcx, args)),
            )
        };

        match &mut impl_trait_ref {
            Some(impl_trait_ref) => {
                assert_eq!(impl_trait_ref.self_ty(), self_ty);
                *impl_trait_ref = self.tcx.normalize_erasing_regions(typing_env, *impl_trait_ref);
                self_ty = impl_trait_ref.self_ty();
            }
            None => {
                self_ty = self.tcx.normalize_erasing_regions(typing_env, self_ty);
            }
        }

        self.default_print_impl_path(impl_def_id, self_ty, impl_trait_ref)
    }
}

impl<'tcx> PrettyPrinter<'tcx> for SymbolPrinter<'tcx> {
    fn should_print_region(&self, _region: ty::Region<'_>) -> bool {
        false
    }
    fn comma_sep<T>(&mut self, mut elems: impl Iterator<Item = T>) -> Result<(), PrintError>
    where
        T: Print<'tcx, Self>,
    {
        if let Some(first) = elems.next() {
            first.print(self)?;
            for elem in elems {
                self.write_str(",")?;
                elem.print(self)?;
            }
        }
        Ok(())
    }

    fn generic_delimiters(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<(), PrintError>,
    ) -> Result<(), PrintError> {
        write!(self, "<")?;

        let kept_within_component = mem::replace(&mut self.keep_within_component, true);
        f(self)?;
        self.keep_within_component = kept_within_component;

        write!(self, ">")?;

        Ok(())
    }
}

impl fmt::Write for SymbolPrinter<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        // Name sanitation. LLVM will happily accept identifiers with weird names, but
        // gas doesn't!
        // gas accepts the following characters in symbols: a-z, A-Z, 0-9, ., _, $
        // NVPTX assembly has more strict naming rules than gas, so additionally, dots
        // are replaced with '$' there.

        for c in s.chars() {
            if self.path.temp_buf.is_empty() {
                match c {
                    'a'..='z' | 'A'..='Z' | '_' => {}
                    _ => {
                        // Underscore-qualify anything that didn't start as an ident.
                        self.path.temp_buf.push('_');
                    }
                }
            }
            match c {
                // Escape these with $ sequences
                '@' => self.path.temp_buf.push_str("$SP$"),
                '*' => self.path.temp_buf.push_str("$BP$"),
                '&' => self.path.temp_buf.push_str("$RF$"),
                '<' => self.path.temp_buf.push_str("$LT$"),
                '>' => self.path.temp_buf.push_str("$GT$"),
                '(' => self.path.temp_buf.push_str("$LP$"),
                ')' => self.path.temp_buf.push_str("$RP$"),
                ',' => self.path.temp_buf.push_str("$C$"),

                '-' | ':' | '.' if self.tcx.has_strict_asm_symbol_naming() => {
                    // NVPTX doesn't support these characters in symbol names.
                    self.path.temp_buf.push('$')
                }

                // '.' doesn't occur in types and functions, so reuse it
                // for ':' and '-'
                '-' | ':' => self.path.temp_buf.push('.'),

                // Avoid crashing LLVM in certain (LTO-related) situations, see #60925.
                'm' if self.path.temp_buf.ends_with(".llv") => self.path.temp_buf.push_str("$u6d$"),

                // These are legal symbols
                'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.' | '$' => self.path.temp_buf.push(c),

                _ => {
                    self.path.temp_buf.push('$');
                    for c in c.escape_unicode().skip(1) {
                        match c {
                            '{' => {}
                            '}' => self.path.temp_buf.push('$'),
                            c => self.path.temp_buf.push(c),
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
