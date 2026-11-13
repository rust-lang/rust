pub use ffi::Blame;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_target::spec::SanitizerSet;

pub(crate) mod ffi;

#[inline]
pub fn is_blame_ignored(no_san: Blame, san: Blame) -> bool {
    no_san.is_some() && (san.is_none() || no_san > san)
}

pub struct SanitizerIgnoreList {
    inner: *mut ffi::Opaque,
}

#[derive(Clone, Copy, Debug)]
pub struct InstanceSanitizers {
    pub enabled: SanitizerSet,
    pub ignore_cfi: bool,
    pub ignore_kcfi: bool,
}

impl SanitizerIgnoreList {
    pub fn new(paths: &[String]) -> Result<Self, String> {
        use std::ffi::CString;
        let c_paths: Vec<CString> =
            paths.iter().map(|p| CString::new(p.as_str()).unwrap()).collect();
        let c_ptrs: Vec<*const libc::c_char> = c_paths.iter().map(|c| c.as_ptr()).collect();

        let mut inner = std::ptr::null_mut();
        let err = ffi::build_string(|err| unsafe {
            inner = ffi::LLVMRustSpecialCaseListCreate(c_ptrs.as_ptr(), c_ptrs.len(), err);
        });

        let err = err.unwrap_or_else(|e| format!("utf8 error: {}", e));
        if inner.is_null() { Err(err) } else { Ok(Self { inner }) }
    }

    pub fn in_section_blame(
        &self,
        section: &std::ffi::CStr,
        prefix: &std::ffi::CStr,
        query: &str,
    ) -> (Blame, Blame) {
        let mask = section_to_sanitizer_set(section);
        let mut no_san = Blame::NONE;
        let mut san = Blame::NONE;
        let Ok(query) = std::ffi::CString::new(query) else {
            return (Blame::NONE, Blame::NONE);
        };
        unsafe {
            ffi::LLVMRustSpecialCaseListInSectionBlame(
                self.inner,
                mask.map(|s| s.bits() as u32).unwrap_or(0),
                section.as_ptr(),
                prefix.as_ptr(),
                query.as_ptr(),
                &mut no_san,
                &mut san,
            );
        }
        (no_san, san)
    }

    pub fn instance_blame<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        section: &std::ffi::CStr,
    ) -> (Blame, Blame) {
        let sym_name = tcx.symbol_name(instance).name;
        let span = tcx.def_span(instance.def_id());
        let filename =
            tcx.sess.source_map().span_to_filename(span).prefer_local_unconditionally().to_string();
        let mainfile = tcx
            .sess
            .local_crate_source_file()
            .and_then(|path| path.local_path().map(|p| p.display().to_string()))
            .unwrap_or_default();
        let demangled =
            rustc_middle::ty::print::with_no_trimmed_paths!(tcx.def_path_str(instance.def_id()));

        let mut no_san = Blame::NONE;
        let mut san = Blame::NONE;
        let mut update = |prefix: &std::ffi::CStr, query: &str| {
            let (ns, s) = self.in_section_blame(section, prefix, query);
            no_san = no_san.max(ns);
            san = san.max(s);
        };

        update(c"fun", sym_name);
        update(c"fun", &demangled);
        update(c"src", &filename);
        if !mainfile.is_empty() {
            update(c"mainfile", &mainfile);
        }

        (no_san, san)
    }

    pub fn is_instance_ignored<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        section: &std::ffi::CStr,
    ) -> bool {
        let (no_san, san) = self.instance_blame(tcx, instance, section);
        is_blame_ignored(no_san, san)
    }

    pub fn filter_instance_sanitizers<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        mut enabled: SanitizerSet,
    ) -> InstanceSanitizers {
        let (address_nosan, address_san) = self.instance_blame(tcx, instance, c"address");
        let (kaddress_nosan, kaddress_san) = self.instance_blame(tcx, instance, c"kernel-address");
        let (hwaddress_nosan, hwaddress_san) = self.instance_blame(tcx, instance, c"hwaddress");
        let (khwaddress_nosan, khwaddress_san) =
            self.instance_blame(tcx, instance, c"kernel-hwaddress");

        let ignore_address = is_blame_ignored(address_nosan, address_san);
        let ignore_kernel_address = is_blame_ignored(kaddress_nosan, kaddress_san);

        let ignore_hwaddress = is_blame_ignored(hwaddress_nosan, hwaddress_san);
        let ignore_kernel_hwaddress = is_blame_ignored(khwaddress_nosan, khwaddress_san);

        if enabled.contains(SanitizerSet::ADDRESS) && ignore_address {
            enabled.remove(SanitizerSet::ADDRESS);
        }
        if enabled.contains(SanitizerSet::KERNELADDRESS) && ignore_kernel_address {
            enabled.remove(SanitizerSet::KERNELADDRESS);
        }
        if enabled.contains(SanitizerSet::MEMORY)
            && self.is_instance_ignored(tcx, instance, c"memory")
        {
            enabled.remove(SanitizerSet::MEMORY);
        }
        if enabled.contains(SanitizerSet::THREAD)
            && self.is_instance_ignored(tcx, instance, c"thread")
        {
            enabled.remove(SanitizerSet::THREAD);
        }
        if enabled.contains(SanitizerSet::HWADDRESS) && ignore_hwaddress {
            enabled.remove(SanitizerSet::HWADDRESS);
        }
        if enabled.contains(SanitizerSet::KERNELHWADDRESS) && ignore_kernel_hwaddress {
            enabled.remove(SanitizerSet::KERNELHWADDRESS);
        }
        // FIXME: Add support for filtering SAFESTACK, SHADOWCALLSTACK, MEMTAG, and REALTIME.
        // Note: For REALTIME, `rustc_codegen_llvm::attributes::sanitize_attrs` will also
        // need to check the filtered `enabled` set rather than `tcx.sess.sanitizers()`.

        let ignore_cfi = self.is_instance_ignored(tcx, instance, c"cfi");
        let ignore_kcfi = self.is_instance_ignored(tcx, instance, c"kcfi");

        InstanceSanitizers { enabled, ignore_cfi, ignore_kcfi }
    }

    pub fn contains_prefix(
        &self,
        section: &std::ffi::CStr,
        prefix: &std::ffi::CStr,
        query: &str,
    ) -> bool {
        let (no_san, san) = self.in_section_blame(section, prefix, query);
        is_blame_ignored(no_san, san)
    }
}

impl Drop for SanitizerIgnoreList {
    fn drop(&mut self) {
        unsafe {
            ffi::LLVMRustSpecialCaseListDestroy(self.inner);
        }
    }
}

pub fn typename_for_ignore_list<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
) -> String {
    let inputs: Vec<_> = fn_abi.args.iter().map(|arg| arg.layout.ty).collect();
    let output = fn_abi.ret.layout.ty;
    let mut fn_sig_kind = ty::FnSigKind::default();
    fn_sig_kind = fn_sig_kind.set_safety(rustc_hir::Safety::Safe);
    fn_sig_kind = fn_sig_kind.set_c_variadic(fn_abi.c_variadic);
    let fn_sig = tcx.mk_fn_sig(inputs, output, fn_sig_kind);
    let fn_ptr = Ty::new_fn_ptr(tcx, ty::Binder::dummy(fn_sig));
    ty::print::with_no_trimmed_paths!(fn_ptr.to_string())
}

fn section_to_sanitizer_set(section: &std::ffi::CStr) -> Option<SanitizerSet> {
    match section.to_bytes() {
        b"address" => Some(SanitizerSet::ADDRESS),
        b"kernel-address" | b"kasan" => Some(SanitizerSet::KERNELADDRESS),
        b"memory" => Some(SanitizerSet::MEMORY),
        b"thread" => Some(SanitizerSet::THREAD),
        b"hwaddress" => Some(SanitizerSet::HWADDRESS),
        b"kernel-hwaddress" | b"khwasan" => Some(SanitizerSet::KERNELHWADDRESS),
        b"safestack" | b"safe-stack" => Some(SanitizerSet::SAFESTACK),
        b"shadow-call-stack" | b"shadowcallstack" => Some(SanitizerSet::SHADOWCALLSTACK),
        b"cfi" | b"cfi-icall" => Some(SanitizerSet::CFI),
        b"kcfi" => Some(SanitizerSet::KCFI),
        b"memtag" => Some(SanitizerSet::MEMTAG),
        b"realtime" => Some(SanitizerSet::REALTIME),
        b"leak" => Some(SanitizerSet::LEAK),
        b"dataflow" => Some(SanitizerSet::DATAFLOW),
        _ => None,
    }
}
