use rustc_ast::{ast, MetaItemKind, NestedMetaItem};
use rustc_attr::{list_contains_name, InlineAttr, InstructionSetAttr, OptimizeAttr};
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId, LOCAL_CRATE};
use rustc_hir::{lang_items, weak_lang_items::WEAK_LANG_ITEMS, LangItem};
use rustc_middle::middle::codegen_fn_attrs::{CodegenFnAttrFlags, CodegenFnAttrs};
use rustc_middle::mir::mono::Linkage;
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self as ty, DefIdTree, TyCtxt};
use rustc_session::{lint, parse::feature_err};
use rustc_span::{sym, Span};
use rustc_target::spec::{abi, SanitizerSet};

use crate::target_features::from_target_feature;
use crate::{errors::ExpectedUsedSymbol, target_features::check_target_feature_trait_unsafe};

fn linkage_by_name(tcx: TyCtxt<'_>, def_id: LocalDefId, name: &str) -> Linkage {
    use rustc_middle::mir::mono::Linkage::*;

    // Use the names from src/llvm/docs/LangRef.rst here. Most types are only
    // applicable to variable declarations and may not really make sense for
    // Rust code in the first place but allow them anyway and trust that the
    // user knows what they're doing. Who knows, unanticipated use cases may pop
    // up in the future.
    //
    // ghost, dllimport, dllexport and linkonce_odr_autohide are not supported
    // and don't have to be, LLVM treats them as no-ops.
    match name {
        "appending" => Appending,
        "available_externally" => AvailableExternally,
        "common" => Common,
        "extern_weak" => ExternalWeak,
        "external" => External,
        "internal" => Internal,
        "linkonce" => LinkOnceAny,
        "linkonce_odr" => LinkOnceODR,
        "private" => Private,
        "weak" => WeakAny,
        "weak_odr" => WeakODR,
        _ => tcx.sess.span_fatal(tcx.def_span(def_id), "invalid linkage specified"),
    }
}

fn codegen_fn_attrs(tcx: TyCtxt<'_>, did: DefId) -> CodegenFnAttrs {
    if cfg!(debug_assertions) {
        let def_kind = tcx.def_kind(did);
        assert!(
            def_kind.has_codegen_attrs(),
            "unexpected `def_kind` in `codegen_fn_attrs`: {def_kind:?}",
        );
    }

    let did = did.expect_local();
    let attrs = tcx.hir().attrs(tcx.hir().local_def_id_to_hir_id(did));
    let mut codegen_fn_attrs = CodegenFnAttrs::new();
    if tcx.should_inherit_track_caller(did) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
    }

    let supported_target_features = tcx.supported_target_features(LOCAL_CRATE);

    // In some cases, attribute are only valid on functions, but it's the `check_attr`
    // pass that check that they aren't used anywhere else, rather this module.
    // In these cases, we bail from performing further checks that are only meaningful for
    // functions (such as calling `fn_sig`, which ICEs if given a non-function). We also
    // report a delayed bug, just in case `check_attr` isn't doing its job.
    let validate_fn_only_attr = |attr_sp| -> bool {
        let def_kind = tcx.def_kind(did);
        if let DefKind::Fn | DefKind::AssocFn | DefKind::Variant | DefKind::Ctor(..) = def_kind {
            true
        } else {
            tcx.sess.delay_span_bug(attr_sp, "this attribute can only be applied to functions");
            false
        }
    };

    let mut inline_span = None;
    let mut link_ordinal_span = None;
    let mut no_sanitize_span = None;
    for attr in attrs.iter() {
        if attr.has_name(sym::cold) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::COLD;
        } else if attr.has_name(sym::rustc_allocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR;
        } else if attr.has_name(sym::ffi_returns_twice) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_RETURNS_TWICE;
        } else if attr.has_name(sym::ffi_pure) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_PURE;
        } else if attr.has_name(sym::ffi_const) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::FFI_CONST;
        } else if attr.has_name(sym::rustc_nounwind) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
        } else if attr.has_name(sym::rustc_reallocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::REALLOCATOR;
        } else if attr.has_name(sym::rustc_deallocator) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::DEALLOCATOR;
        } else if attr.has_name(sym::rustc_allocator_zeroed) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::ALLOCATOR_ZEROED;
        } else if attr.has_name(sym::naked) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NAKED;
        } else if attr.has_name(sym::no_mangle) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
        } else if attr.has_name(sym::no_coverage) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_COVERAGE;
        } else if attr.has_name(sym::rustc_std_internal_symbol) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
        } else if attr.has_name(sym::used) {
            let inner = attr.meta_item_list();
            match inner.as_deref() {
                Some([item]) if item.has_name(sym::linker) => {
                    if !tcx.features().used_with_arg {
                        feature_err(
                            &tcx.sess.parse_sess,
                            sym::used_with_arg,
                            attr.span,
                            "`#[used(linker)]` is currently unstable",
                        )
                        .emit();
                    }
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED_LINKER;
                }
                Some([item]) if item.has_name(sym::compiler) => {
                    if !tcx.features().used_with_arg {
                        feature_err(
                            &tcx.sess.parse_sess,
                            sym::used_with_arg,
                            attr.span,
                            "`#[used(compiler)]` is currently unstable",
                        )
                        .emit();
                    }
                    codegen_fn_attrs.flags |= CodegenFnAttrFlags::USED;
                }
                Some(_) => {
                    tcx.sess.emit_err(ExpectedUsedSymbol { span: attr.span });
                }
                None => {
                    // Unfortunately, unconditionally using `llvm.used` causes
                    // issues in handling `.init_array` with the gold linker,
                    // but using `llvm.compiler.used` caused a nontrival amount
                    // of unintentional ecosystem breakage -- particularly on
                    // Mach-O targets.
                    //
                    // As a result, we emit `llvm.compiler.used` only on ELF
                    // targets. This is somewhat ad-hoc, but actually follows
                    // our pre-LLVM 13 behavior (prior to the ecosystem
                    // breakage), and seems to match `clang`'s behavior as well
                    // (both before and after LLVM 13), possibly because they
                    // have similar compatibility concerns to us. See
                    // https://github.com/rust-lang/rust/issues/47384#issuecomment-1019080146
                    // and following comments for some discussion of this, as
                    // well as the comments in `rustc_codegen_llvm` where these
                    // flags are handled.
                    //
                    // Anyway, to be clear: this is still up in the air
                    // somewhat, and is subject to change in the future (which
                    // is a good thing, because this would ideally be a bit
                    // more firmed up).
                    let is_like_elf = !(tcx.sess.target.is_like_osx
                        || tcx.sess.target.is_like_windows
                        || tcx.sess.target.is_like_wasm);
                    codegen_fn_attrs.flags |= if is_like_elf {
                        CodegenFnAttrFlags::USED
                    } else {
                        CodegenFnAttrFlags::USED_LINKER
                    };
                }
            }
        } else if attr.has_name(sym::cmse_nonsecure_entry) {
            if validate_fn_only_attr(attr.span)
                && !matches!(tcx.fn_sig(did).skip_binder().abi(), abi::Abi::C { .. })
            {
                struct_span_err!(
                    tcx.sess,
                    attr.span,
                    E0776,
                    "`#[cmse_nonsecure_entry]` requires C ABI"
                )
                .emit();
            }
            if !tcx.sess.target.llvm_target.contains("thumbv8m") {
                struct_span_err!(tcx.sess, attr.span, E0775, "`#[cmse_nonsecure_entry]` is only valid for targets with the TrustZone-M extension")
                    .emit();
            }
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::CMSE_NONSECURE_ENTRY;
        } else if attr.has_name(sym::thread_local) {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::THREAD_LOCAL;
        } else if attr.has_name(sym::track_caller) {
            if !tcx.is_closure(did.to_def_id())
                && validate_fn_only_attr(attr.span)
                && tcx.fn_sig(did).skip_binder().abi() != abi::Abi::Rust
            {
                struct_span_err!(tcx.sess, attr.span, E0737, "`#[track_caller]` requires Rust ABI")
                    .emit();
            }
            if tcx.is_closure(did.to_def_id()) && !tcx.features().closure_track_caller {
                feature_err(
                    &tcx.sess.parse_sess,
                    sym::closure_track_caller,
                    attr.span,
                    "`#[track_caller]` on closures is currently unstable",
                )
                .emit();
            }
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::TRACK_CALLER;
        } else if attr.has_name(sym::export_name) {
            if let Some(s) = attr.value_str() {
                if s.as_str().contains('\0') {
                    // `#[export_name = ...]` will be converted to a null-terminated string,
                    // so it may not contain any null characters.
                    struct_span_err!(
                        tcx.sess,
                        attr.span,
                        E0648,
                        "`export_name` may not contain null characters"
                    )
                    .emit();
                }
                codegen_fn_attrs.export_name = Some(s);
            }
        } else if attr.has_name(sym::target_feature) {
            if !tcx.is_closure(did.to_def_id())
                && tcx.fn_sig(did).skip_binder().unsafety() == hir::Unsafety::Normal
            {
                // The `#[target_feature]` attribute is allowed on
                // WebAssembly targets on all functions, including safe
                // ones. Other targets have conditions on the usage of
                // `#[target_feature]` because on most targets
                // execution of instructions that are not supported is
                // considered undefined behavior. For WebAssembly which is a
                // 100% safe target at execution time it's not possible to
                // execute undefined instructions, and even if a future
                // feature was added in some form for this it would be a
                // deterministic trap. There is no undefined behavior when
                // executing WebAssembly so `#[target_feature]` is allowed
                // on safe functions (but again, only for WebAssembly)
                //
                // Note that this is also allowed if `actually_rustdoc` so
                // if a target is documenting some wasm-specific code then
                // it's not spuriously denied.
                if !(tcx.sess.target.is_like_wasm || tcx.sess.opts.actually_rustdoc) {
                    check_target_feature_trait_unsafe(tcx, did, attr.span);
                }
            }
            from_target_feature(
                tcx,
                attr,
                supported_target_features,
                &mut codegen_fn_attrs.target_features,
            );
        } else if attr.has_name(sym::linkage) {
            if let Some(val) = attr.value_str() {
                let linkage = Some(linkage_by_name(tcx, did, val.as_str()));
                if tcx.is_foreign_item(did) {
                    codegen_fn_attrs.import_linkage = linkage;
                } else {
                    codegen_fn_attrs.linkage = linkage;
                }
            }
        } else if attr.has_name(sym::link_section) {
            if let Some(val) = attr.value_str() {
                if val.as_str().bytes().any(|b| b == 0) {
                    let msg = format!(
                        "illegal null byte in link_section \
                         value: `{}`",
                        &val
                    );
                    tcx.sess.span_err(attr.span, &msg);
                } else {
                    codegen_fn_attrs.link_section = Some(val);
                }
            }
        } else if attr.has_name(sym::link_name) {
            codegen_fn_attrs.link_name = attr.value_str();
        } else if attr.has_name(sym::link_ordinal) {
            link_ordinal_span = Some(attr.span);
            if let ordinal @ Some(_) = check_link_ordinal(tcx, attr) {
                codegen_fn_attrs.link_ordinal = ordinal;
            }
        } else if attr.has_name(sym::no_sanitize) {
            no_sanitize_span = Some(attr.span);
            if let Some(list) = attr.meta_item_list() {
                for item in list.iter() {
                    if item.has_name(sym::address) {
                        codegen_fn_attrs.no_sanitize |=
                            SanitizerSet::ADDRESS | SanitizerSet::KERNELADDRESS;
                    } else if item.has_name(sym::cfi) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::CFI;
                    } else if item.has_name(sym::kcfi) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::KCFI;
                    } else if item.has_name(sym::memory) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMORY;
                    } else if item.has_name(sym::memtag) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::MEMTAG;
                    } else if item.has_name(sym::shadow_call_stack) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::SHADOWCALLSTACK;
                    } else if item.has_name(sym::thread) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::THREAD;
                    } else if item.has_name(sym::hwaddress) {
                        codegen_fn_attrs.no_sanitize |= SanitizerSet::HWADDRESS;
                    } else {
                        tcx.sess
                            .struct_span_err(item.span(), "invalid argument for `no_sanitize`")
                            .note("expected one of: `address`, `cfi`, `hwaddress`, `kcfi`, `memory`, `memtag`, `shadow-call-stack`, or `thread`")
                            .emit();
                    }
                }
            }
        } else if attr.has_name(sym::instruction_set) {
            codegen_fn_attrs.instruction_set = attr.meta_item_list().and_then(|l| match &l[..] {
                [NestedMetaItem::MetaItem(set)] => {
                    let segments =
                        set.path.segments.iter().map(|x| x.ident.name).collect::<Vec<_>>();
                    match segments.as_slice() {
                        [sym::arm, sym::a32] | [sym::arm, sym::t32] => {
                            if !tcx.sess.target.has_thumb_interworking {
                                struct_span_err!(
                                    tcx.sess.diagnostic(),
                                    attr.span,
                                    E0779,
                                    "target does not support `#[instruction_set]`"
                                )
                                .emit();
                                None
                            } else if segments[1] == sym::a32 {
                                Some(InstructionSetAttr::ArmA32)
                            } else if segments[1] == sym::t32 {
                                Some(InstructionSetAttr::ArmT32)
                            } else {
                                unreachable!()
                            }
                        }
                        _ => {
                            struct_span_err!(
                                tcx.sess.diagnostic(),
                                attr.span,
                                E0779,
                                "invalid instruction set specified",
                            )
                            .emit();
                            None
                        }
                    }
                }
                [] => {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0778,
                        "`#[instruction_set]` requires an argument"
                    )
                    .emit();
                    None
                }
                _ => {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0779,
                        "cannot specify more than one instruction set"
                    )
                    .emit();
                    None
                }
            })
        } else if attr.has_name(sym::repr) {
            codegen_fn_attrs.alignment = match attr.meta_item_list() {
                Some(items) => match items.as_slice() {
                    [item] => match item.name_value_literal() {
                        Some((sym::align, literal)) => {
                            let alignment = rustc_attr::parse_alignment(&literal.kind);

                            match alignment {
                                Ok(align) => Some(align),
                                Err(msg) => {
                                    struct_span_err!(
                                        tcx.sess.diagnostic(),
                                        attr.span,
                                        E0589,
                                        "invalid `repr(align)` attribute: {}",
                                        msg
                                    )
                                    .emit();

                                    None
                                }
                            }
                        }
                        _ => None,
                    },
                    [] => None,
                    _ => None,
                },
                None => None,
            };
        }
    }

    codegen_fn_attrs.inline = attrs.iter().fold(InlineAttr::None, |ia, attr| {
        if !attr.has_name(sym::inline) {
            return ia;
        }
        match attr.meta_kind() {
            Some(MetaItemKind::Word) => InlineAttr::Hint,
            Some(MetaItemKind::List(ref items)) => {
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        attr.span,
                        E0534,
                        "expected one argument"
                    )
                    .emit();
                    InlineAttr::None
                } else if list_contains_name(&items, sym::always) {
                    InlineAttr::Always
                } else if list_contains_name(&items, sym::never) {
                    InlineAttr::Never
                } else {
                    struct_span_err!(
                        tcx.sess.diagnostic(),
                        items[0].span(),
                        E0535,
                        "invalid argument"
                    )
                    .help("valid inline arguments are `always` and `never`")
                    .emit();

                    InlineAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    codegen_fn_attrs.optimize = attrs.iter().fold(OptimizeAttr::None, |ia, attr| {
        if !attr.has_name(sym::optimize) {
            return ia;
        }
        let err = |sp, s| struct_span_err!(tcx.sess.diagnostic(), sp, E0722, "{}", s).emit();
        match attr.meta_kind() {
            Some(MetaItemKind::Word) => {
                err(attr.span, "expected one argument");
                ia
            }
            Some(MetaItemKind::List(ref items)) => {
                inline_span = Some(attr.span);
                if items.len() != 1 {
                    err(attr.span, "expected one argument");
                    OptimizeAttr::None
                } else if list_contains_name(&items, sym::size) {
                    OptimizeAttr::Size
                } else if list_contains_name(&items, sym::speed) {
                    OptimizeAttr::Speed
                } else {
                    err(items[0].span(), "invalid argument");
                    OptimizeAttr::None
                }
            }
            Some(MetaItemKind::NameValue(_)) => ia,
            None => ia,
        }
    });

    // #73631: closures inherit `#[target_feature]` annotations
    if tcx.is_closure(did.to_def_id()) {
        let owner_id = tcx.parent(did.to_def_id());
        if tcx.def_kind(owner_id).has_codegen_attrs() {
            codegen_fn_attrs
                .target_features
                .extend(tcx.codegen_fn_attrs(owner_id).target_features.iter().copied());
        }
    }

    // If a function uses #[target_feature] it can't be inlined into general
    // purpose functions as they wouldn't have the right target features
    // enabled. For that reason we also forbid #[inline(always)] as it can't be
    // respected.
    if !codegen_fn_attrs.target_features.is_empty() {
        if codegen_fn_attrs.inline == InlineAttr::Always {
            if let Some(span) = inline_span {
                tcx.sess.span_err(
                    span,
                    "cannot use `#[inline(always)]` with \
                     `#[target_feature]`",
                );
            }
        }
    }

    if !codegen_fn_attrs.no_sanitize.is_empty() {
        if codegen_fn_attrs.inline == InlineAttr::Always {
            if let (Some(no_sanitize_span), Some(inline_span)) = (no_sanitize_span, inline_span) {
                let hir_id = tcx.hir().local_def_id_to_hir_id(did);
                tcx.struct_span_lint_hir(
                    lint::builtin::INLINE_NO_SANITIZE,
                    hir_id,
                    no_sanitize_span,
                    "`no_sanitize` will have no effect after inlining",
                    |lint| lint.span_note(inline_span, "inlining requested here"),
                )
            }
        }
    }

    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::NAKED) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_COVERAGE;
        codegen_fn_attrs.inline = InlineAttr::Never;
    }

    // Weak lang items have the same semantics as "std internal" symbols in the
    // sense that they're preserved through all our LTO passes and only
    // strippable by the linker.
    //
    // Additionally weak lang items have predetermined symbol names.
    if WEAK_LANG_ITEMS.iter().any(|&l| tcx.lang_items().get(l) == Some(did.to_def_id())) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL;
    }
    if let Some((name, _)) = lang_items::extract(attrs)
        && let Some(lang_item) = LangItem::from_name(name)
        && let Some(link_name) = lang_item.link_name()
    {
        codegen_fn_attrs.export_name = Some(link_name);
        codegen_fn_attrs.link_name = Some(link_name);
    }
    check_link_name_xor_ordinal(tcx, &codegen_fn_attrs, link_ordinal_span);

    // Internal symbols to the standard library all have no_mangle semantics in
    // that they have defined symbol names present in the function name. This
    // also applies to weak symbols where they all have known symbol names.
    if codegen_fn_attrs.flags.contains(CodegenFnAttrFlags::RUSTC_STD_INTERNAL_SYMBOL) {
        codegen_fn_attrs.flags |= CodegenFnAttrFlags::NO_MANGLE;
    }

    // Any linkage to LLVM intrinsics for now forcibly marks them all as never
    // unwinds since LLVM sometimes can't handle codegen which `invoke`s
    // intrinsic functions.
    if let Some(name) = &codegen_fn_attrs.link_name {
        if name.as_str().starts_with("llvm.") {
            codegen_fn_attrs.flags |= CodegenFnAttrFlags::NEVER_UNWIND;
        }
    }

    codegen_fn_attrs
}

/// Checks if the provided DefId is a method in a trait impl for a trait which has track_caller
/// applied to the method prototype.
fn should_inherit_track_caller(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    if let Some(impl_item) = tcx.opt_associated_item(def_id)
        && let ty::AssocItemContainer::ImplContainer = impl_item.container
        && let Some(trait_item) = impl_item.trait_item_def_id
    {
        return tcx
            .codegen_fn_attrs(trait_item)
            .flags
            .intersects(CodegenFnAttrFlags::TRACK_CALLER);
    }

    false
}

fn check_link_ordinal(tcx: TyCtxt<'_>, attr: &ast::Attribute) -> Option<u16> {
    use rustc_ast::{LitIntType, LitKind, MetaItemLit};
    if !tcx.features().raw_dylib && tcx.sess.target.arch == "x86" {
        feature_err(
            &tcx.sess.parse_sess,
            sym::raw_dylib,
            attr.span,
            "`#[link_ordinal]` is unstable on x86",
        )
        .emit();
    }
    let meta_item_list = attr.meta_item_list();
    let meta_item_list = meta_item_list.as_deref();
    let sole_meta_list = match meta_item_list {
        Some([item]) => item.lit(),
        Some(_) => {
            tcx.sess
                .struct_span_err(attr.span, "incorrect number of arguments to `#[link_ordinal]`")
                .note("the attribute requires exactly one argument")
                .emit();
            return None;
        }
        _ => None,
    };
    if let Some(MetaItemLit { kind: LitKind::Int(ordinal, LitIntType::Unsuffixed), .. }) =
        sole_meta_list
    {
        // According to the table at https://docs.microsoft.com/en-us/windows/win32/debug/pe-format#import-header,
        // the ordinal must fit into 16 bits. Similarly, the Ordinal field in COFFShortExport (defined
        // in llvm/include/llvm/Object/COFFImportFile.h), which we use to communicate import information
        // to LLVM for `#[link(kind = "raw-dylib"_])`, is also defined to be uint16_t.
        //
        // FIXME: should we allow an ordinal of 0?  The MSVC toolchain has inconsistent support for this:
        // both LINK.EXE and LIB.EXE signal errors and abort when given a .DEF file that specifies
        // a zero ordinal. However, llvm-dlltool is perfectly happy to generate an import library
        // for such a .DEF file, and MSVC's LINK.EXE is also perfectly happy to consume an import
        // library produced by LLVM with an ordinal of 0, and it generates an .EXE.  (I don't know yet
        // if the resulting EXE runs, as I haven't yet built the necessary DLL -- see earlier comment
        // about LINK.EXE failing.)
        if *ordinal <= u16::MAX as u128 {
            Some(*ordinal as u16)
        } else {
            let msg = format!("ordinal value in `link_ordinal` is too large: `{}`", &ordinal);
            tcx.sess
                .struct_span_err(attr.span, &msg)
                .note("the value may not exceed `u16::MAX`")
                .emit();
            None
        }
    } else {
        tcx.sess
            .struct_span_err(attr.span, "illegal ordinal format in `link_ordinal`")
            .note("an unsuffixed integer value, e.g., `1`, is expected")
            .emit();
        None
    }
}

fn check_link_name_xor_ordinal(
    tcx: TyCtxt<'_>,
    codegen_fn_attrs: &CodegenFnAttrs,
    inline_span: Option<Span>,
) {
    if codegen_fn_attrs.link_name.is_none() || codegen_fn_attrs.link_ordinal.is_none() {
        return;
    }
    let msg = "cannot use `#[link_name]` with `#[link_ordinal]`";
    if let Some(span) = inline_span {
        tcx.sess.span_err(span, msg);
    } else {
        tcx.sess.err(msg);
    }
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { codegen_fn_attrs, should_inherit_track_caller, ..*providers };
}
