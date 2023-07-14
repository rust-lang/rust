//! Interpret intrinsics, lang items and `extern "C"` wellknown functions which their implementation
//! is not available.

use std::cmp;

use chalk_ir::TyKind;
use hir_def::resolver::HasResolver;
use hir_expand::mod_path::ModPath;

use super::*;

mod simd;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(it) => it,
            Err(_) => return Err(MirEvalError::TypeError("mismatched size")),
        }))
    };
}

macro_rules! not_supported {
    ($it: expr) => {
        return Err(MirEvalError::NotSupported(format!($it)))
    };
}

impl Evaluator<'_> {
    pub(super) fn detect_and_exec_special_function(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        locals: &Locals,
        destination: Interval,
        span: MirSpan,
    ) -> Result<bool> {
        let function_data = self.db.function_data(def);
        let is_intrinsic = match &function_data.abi {
            Some(abi) => *abi == Interned::new_str("rust-intrinsic"),
            None => match def.lookup(self.db.upcast()).container {
                hir_def::ItemContainerId::ExternBlockId(block) => {
                    let id = block.lookup(self.db.upcast()).id;
                    id.item_tree(self.db.upcast())[id.value].abi.as_deref()
                        == Some("rust-intrinsic")
                }
                _ => false,
            },
        };
        if is_intrinsic {
            self.exec_intrinsic(
                function_data.name.as_text().unwrap_or_default().as_str(),
                args,
                generic_args,
                destination,
                &locals,
                span,
            )?;
            return Ok(true);
        }
        let is_platform_intrinsic = match &function_data.abi {
            Some(abi) => *abi == Interned::new_str("platform-intrinsic"),
            None => match def.lookup(self.db.upcast()).container {
                hir_def::ItemContainerId::ExternBlockId(block) => {
                    let id = block.lookup(self.db.upcast()).id;
                    id.item_tree(self.db.upcast())[id.value].abi.as_deref()
                        == Some("platform-intrinsic")
                }
                _ => false,
            },
        };
        if is_platform_intrinsic {
            self.exec_platform_intrinsic(
                function_data.name.as_text().unwrap_or_default().as_str(),
                args,
                generic_args,
                destination,
                &locals,
                span,
            )?;
            return Ok(true);
        }
        let is_extern_c = match def.lookup(self.db.upcast()).container {
            hir_def::ItemContainerId::ExternBlockId(block) => {
                let id = block.lookup(self.db.upcast()).id;
                id.item_tree(self.db.upcast())[id.value].abi.as_deref() == Some("C")
            }
            _ => false,
        };
        if is_extern_c {
            self.exec_extern_c(
                function_data.name.as_text().unwrap_or_default().as_str(),
                args,
                generic_args,
                destination,
                &locals,
                span,
            )?;
            return Ok(true);
        }
        let alloc_fn = function_data
            .attrs
            .iter()
            .filter_map(|it| it.path().as_ident())
            .filter_map(|it| it.as_str())
            .find(|it| {
                [
                    "rustc_allocator",
                    "rustc_deallocator",
                    "rustc_reallocator",
                    "rustc_allocator_zeroed",
                ]
                .contains(it)
            });
        if let Some(alloc_fn) = alloc_fn {
            self.exec_alloc_fn(alloc_fn, args, destination)?;
            return Ok(true);
        }
        if let Some(it) = self.detect_lang_function(def) {
            let arg_bytes =
                args.iter().map(|it| Ok(it.get(&self)?.to_owned())).collect::<Result<Vec<_>>>()?;
            let result = self.exec_lang_item(it, generic_args, &arg_bytes, locals, span)?;
            destination.write_from_bytes(self, &result)?;
            return Ok(true);
        }
        Ok(false)
    }

    fn exec_alloc_fn(
        &mut self,
        alloc_fn: &str,
        args: &[IntervalAndTy],
        destination: Interval,
    ) -> Result<()> {
        match alloc_fn {
            "rustc_allocator_zeroed" | "rustc_allocator" => {
                let [size, align] = args else {
                    return Err(MirEvalError::TypeError("rustc_allocator args are not provided"));
                };
                let size = from_bytes!(usize, size.get(self)?);
                let align = from_bytes!(usize, align.get(self)?);
                let result = self.heap_allocate(size, align)?;
                destination.write_from_bytes(self, &result.to_bytes())?;
            }
            "rustc_deallocator" => { /* no-op for now */ }
            "rustc_reallocator" => {
                let [ptr, old_size, align, new_size] = args else {
                    return Err(MirEvalError::TypeError("rustc_allocator args are not provided"));
                };
                let old_size = from_bytes!(usize, old_size.get(self)?);
                let new_size = from_bytes!(usize, new_size.get(self)?);
                if old_size >= new_size {
                    destination.write_from_interval(self, ptr.interval)?;
                } else {
                    let ptr = Address::from_bytes(ptr.get(self)?)?;
                    let align = from_bytes!(usize, align.get(self)?);
                    let result = self.heap_allocate(new_size, align)?;
                    Interval { addr: result, size: old_size }
                        .write_from_interval(self, Interval { addr: ptr, size: old_size })?;
                    destination.write_from_bytes(self, &result.to_bytes())?;
                }
            }
            _ => not_supported!("unknown alloc function"),
        }
        Ok(())
    }

    fn detect_lang_function(&self, def: FunctionId) -> Option<LangItem> {
        use LangItem::*;
        let candidate = self.db.lang_attr(def.into())?;
        // We want to execute these functions with special logic
        if [PanicFmt, BeginPanic, SliceLen, DropInPlace].contains(&candidate) {
            return Some(candidate);
        }
        None
    }

    fn exec_lang_item(
        &mut self,
        it: LangItem,
        generic_args: &Substitution,
        args: &[Vec<u8>],
        locals: &Locals,
        span: MirSpan,
    ) -> Result<Vec<u8>> {
        use LangItem::*;
        let mut args = args.iter();
        match it {
            BeginPanic => Err(MirEvalError::Panic("<unknown-panic-payload>".to_string())),
            PanicFmt => {
                let message = (|| {
                    let resolver = self.db.crate_def_map(self.crate_id).crate_root().resolver(self.db.upcast());
                    let Some(format_fn) = resolver.resolve_path_in_value_ns_fully(
                        self.db.upcast(),
                        &hir_def::path::Path::from_known_path_with_no_generic(ModPath::from_segments(
                            hir_expand::mod_path::PathKind::Abs,
                            [name![std], name![fmt], name![format]].into_iter(),
                        )),
                    ) else {
                        not_supported!("std::fmt::format not found");
                    };
                    let hir_def::resolver::ValueNs::FunctionId(format_fn) = format_fn else { not_supported!("std::fmt::format is not a function") };
                    let message_string = self.interpret_mir(self.db.mir_body(format_fn.into()).map_err(|e| MirEvalError::MirLowerError(format_fn, e))?, args.map(|x| IntervalOrOwned::Owned(x.clone())))?;
                    let addr = Address::from_bytes(&message_string[self.ptr_size()..2 * self.ptr_size()])?;
                    let size = from_bytes!(usize, message_string[2 * self.ptr_size()..]);
                    Ok(std::string::String::from_utf8_lossy(self.read_memory(addr, size)?).into_owned())
                })()
                .unwrap_or_else(|e| format!("Failed to render panic format args: {e:?}"));
                Err(MirEvalError::Panic(message))
            }
            SliceLen => {
                let arg = args
                    .next()
                    .ok_or(MirEvalError::TypeError("argument of <[T]>::len() is not provided"))?;
                let ptr_size = arg.len() / 2;
                Ok(arg[ptr_size..].into())
            }
            DropInPlace => {
                let ty =
                    generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner)).ok_or(
                        MirEvalError::TypeError(
                            "generic argument of drop_in_place is not provided",
                        ),
                    )?;
                let arg = args
                    .next()
                    .ok_or(MirEvalError::TypeError("argument of drop_in_place is not provided"))?;
                self.run_drop_glue_deep(
                    ty.clone(),
                    locals,
                    Address::from_bytes(&arg[0..self.ptr_size()])?,
                    &arg[self.ptr_size()..],
                    span,
                )?;
                Ok(vec![])
            }
            it => not_supported!("Executing lang item {it:?}"),
        }
    }

    fn exec_syscall(
        &mut self,
        id: i64,
        args: &[IntervalAndTy],
        destination: Interval,
        _locals: &Locals,
        _span: MirSpan,
    ) -> Result<()> {
        match id {
            318 => {
                // SYS_getrandom
                let [buf, len, _flags] = args else {
                    return Err(MirEvalError::TypeError("SYS_getrandom args are not provided"));
                };
                let addr = Address::from_bytes(buf.get(self)?)?;
                let size = from_bytes!(usize, len.get(self)?);
                for i in 0..size {
                    let rand_byte = self.random_state.rand_u64() as u8;
                    self.write_memory(addr.offset(i), &[rand_byte])?;
                }
                destination.write_from_interval(self, len.interval)
            }
            _ => {
                not_supported!("Unknown syscall id {id:?}")
            }
        }
    }

    fn exec_extern_c(
        &mut self,
        as_str: &str,
        args: &[IntervalAndTy],
        _generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        span: MirSpan,
    ) -> Result<()> {
        match as_str {
            "memcmp" => {
                let [ptr1, ptr2, size] = args else {
                    return Err(MirEvalError::TypeError("memcmp args are not provided"));
                };
                let addr1 = Address::from_bytes(ptr1.get(self)?)?;
                let addr2 = Address::from_bytes(ptr2.get(self)?)?;
                let size = from_bytes!(usize, size.get(self)?);
                let slice1 = self.read_memory(addr1, size)?;
                let slice2 = self.read_memory(addr2, size)?;
                let r: i128 = match slice1.cmp(slice2) {
                    cmp::Ordering::Less => -1,
                    cmp::Ordering::Equal => 0,
                    cmp::Ordering::Greater => 1,
                };
                destination.write_from_bytes(self, &r.to_le_bytes()[..destination.size])
            }
            "write" => {
                let [fd, ptr, len] = args else {
                    return Err(MirEvalError::TypeError("libc::write args are not provided"));
                };
                let fd = u128::from_le_bytes(pad16(fd.get(self)?, false));
                let interval = Interval {
                    addr: Address::from_bytes(ptr.get(self)?)?,
                    size: from_bytes!(usize, len.get(self)?),
                };
                match fd {
                    1 => {
                        self.write_to_stdout(interval)?;
                    }
                    2 => {
                        self.write_to_stderr(interval)?;
                    }
                    _ => not_supported!("write to arbitrary file descriptor"),
                }
                destination.write_from_interval(self, len.interval)?;
                Ok(())
            }
            "pthread_key_create" => {
                let key = self.thread_local_storage.create_key();
                let Some(arg0) = args.get(0) else {
                    return Err(MirEvalError::TypeError("pthread_key_create arg0 is not provided"));
                };
                let arg0_addr = Address::from_bytes(arg0.get(self)?)?;
                let key_ty = if let Some((ty, ..)) = arg0.ty.as_reference_or_ptr() {
                    ty
                } else {
                    return Err(MirEvalError::TypeError(
                        "pthread_key_create arg0 is not a pointer",
                    ));
                };
                let arg0_interval = Interval::new(
                    arg0_addr,
                    self.size_of_sized(key_ty, locals, "pthread_key_create key arg")?,
                );
                arg0_interval.write_from_bytes(self, &key.to_le_bytes()[0..arg0_interval.size])?;
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_getspecific" => {
                let Some(arg0) = args.get(0) else {
                    return Err(MirEvalError::TypeError(
                        "pthread_getspecific arg0 is not provided",
                    ));
                };
                let key = from_bytes!(usize, &pad16(arg0.get(self)?, false)[0..8]);
                let value = self.thread_local_storage.get_key(key)?;
                destination.write_from_bytes(self, &value.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_setspecific" => {
                let Some(arg0) = args.get(0) else {
                    return Err(MirEvalError::TypeError(
                        "pthread_setspecific arg0 is not provided",
                    ));
                };
                let key = from_bytes!(usize, &pad16(arg0.get(self)?, false)[0..8]);
                let Some(arg1) = args.get(1) else {
                    return Err(MirEvalError::TypeError(
                        "pthread_setspecific arg1 is not provided",
                    ));
                };
                let value = from_bytes!(u128, pad16(arg1.get(self)?, false));
                self.thread_local_storage.set_key(key, value)?;
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "pthread_key_delete" => {
                // we ignore this currently
                // return 0 as success
                destination.write_from_bytes(self, &0u64.to_le_bytes()[0..destination.size])?;
                Ok(())
            }
            "syscall" => {
                let Some((id, rest)) = args.split_first() else {
                    return Err(MirEvalError::TypeError(
                        "syscall arg1 is not provided",
                    ));
                };
                let id = from_bytes!(i64, id.get(self)?);
                self.exec_syscall(id, rest, destination, locals, span)
            }
            "sched_getaffinity" => {
                let [_pid, _set_size, set] = args else {
                    return Err(MirEvalError::TypeError("libc::write args are not provided"));
                };
                let set = Address::from_bytes(set.get(self)?)?;
                // Only enable core 0 (we are single threaded anyway), which is bitset 0x0000001
                self.write_memory(set, &[1])?;
                // return 0 as success
                self.write_memory_using_ref(destination.addr, destination.size)?.fill(0);
                Ok(())
            }
            _ => not_supported!("unknown external function {as_str}"),
        }
    }

    fn exec_platform_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        span: MirSpan,
    ) -> Result<()> {
        if let Some(name) = name.strip_prefix("simd_") {
            return self.exec_simd_intrinsic(name, args, generic_args, destination, locals, span);
        }
        not_supported!("unknown platform intrinsic {name}");
    }

    fn exec_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        span: MirSpan,
    ) -> Result<()> {
        if let Some(name) = name.strip_prefix("atomic_") {
            return self.exec_atomic_intrinsic(name, args, generic_args, destination, locals, span);
        }
        if let Some(name) = name.strip_suffix("f64") {
            let result = match name {
                "sqrt" | "sin" | "cos" | "exp" | "exp2" | "log" | "log10" | "log2" | "fabs"
                | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" | "roundeven" => {
                    let [arg] = args else {
                        return Err(MirEvalError::TypeError(
                            "f64 intrinsic signature doesn't match fn (f64) -> f64",
                        ));
                    };
                    let arg = from_bytes!(f64, arg.get(self)?);
                    match name {
                        "sqrt" => arg.sqrt(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "exp" => arg.exp(),
                        "exp2" => arg.exp2(),
                        "log" => arg.ln(),
                        "log10" => arg.log10(),
                        "log2" => arg.log2(),
                        "fabs" => arg.abs(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "trunc" => arg.trunc(),
                        // FIXME: these rounds should be different, but only `.round()` is stable now.
                        "rint" => arg.round(),
                        "nearbyint" => arg.round(),
                        "round" => arg.round(),
                        "roundeven" => arg.round(),
                        _ => unreachable!(),
                    }
                }
                "pow" | "minnum" | "maxnum" | "copysign" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::TypeError(
                            "f64 intrinsic signature doesn't match fn (f64, f64) -> f64",
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(f64, arg2.get(self)?);
                    match name {
                        "pow" => arg1.powf(arg2),
                        "minnum" => arg1.min(arg2),
                        "maxnum" => arg1.max(arg2),
                        "copysign" => arg1.copysign(arg2),
                        _ => unreachable!(),
                    }
                }
                "powi" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::TypeError(
                            "powif64 signature doesn't match fn (f64, i32) -> f64",
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(i32, arg2.get(self)?);
                    arg1.powi(arg2)
                }
                "fma" => {
                    let [arg1, arg2, arg3] = args else {
                        return Err(MirEvalError::TypeError(
                            "fmaf64 signature doesn't match fn (f64, f64, f64) -> f64",
                        ));
                    };
                    let arg1 = from_bytes!(f64, arg1.get(self)?);
                    let arg2 = from_bytes!(f64, arg2.get(self)?);
                    let arg3 = from_bytes!(f64, arg3.get(self)?);
                    arg1.mul_add(arg2, arg3)
                }
                _ => not_supported!("unknown f64 intrinsic {name}"),
            };
            return destination.write_from_bytes(self, &result.to_le_bytes());
        }
        if let Some(name) = name.strip_suffix("f32") {
            let result = match name {
                "sqrt" | "sin" | "cos" | "exp" | "exp2" | "log" | "log10" | "log2" | "fabs"
                | "floor" | "ceil" | "trunc" | "rint" | "nearbyint" | "round" | "roundeven" => {
                    let [arg] = args else {
                        return Err(MirEvalError::TypeError(
                            "f32 intrinsic signature doesn't match fn (f32) -> f32",
                        ));
                    };
                    let arg = from_bytes!(f32, arg.get(self)?);
                    match name {
                        "sqrt" => arg.sqrt(),
                        "sin" => arg.sin(),
                        "cos" => arg.cos(),
                        "exp" => arg.exp(),
                        "exp2" => arg.exp2(),
                        "log" => arg.ln(),
                        "log10" => arg.log10(),
                        "log2" => arg.log2(),
                        "fabs" => arg.abs(),
                        "floor" => arg.floor(),
                        "ceil" => arg.ceil(),
                        "trunc" => arg.trunc(),
                        // FIXME: these rounds should be different, but only `.round()` is stable now.
                        "rint" => arg.round(),
                        "nearbyint" => arg.round(),
                        "round" => arg.round(),
                        "roundeven" => arg.round(),
                        _ => unreachable!(),
                    }
                }
                "pow" | "minnum" | "maxnum" | "copysign" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::TypeError(
                            "f32 intrinsic signature doesn't match fn (f32, f32) -> f32",
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(f32, arg2.get(self)?);
                    match name {
                        "pow" => arg1.powf(arg2),
                        "minnum" => arg1.min(arg2),
                        "maxnum" => arg1.max(arg2),
                        "copysign" => arg1.copysign(arg2),
                        _ => unreachable!(),
                    }
                }
                "powi" => {
                    let [arg1, arg2] = args else {
                        return Err(MirEvalError::TypeError(
                            "powif32 signature doesn't match fn (f32, i32) -> f32",
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(i32, arg2.get(self)?);
                    arg1.powi(arg2)
                }
                "fma" => {
                    let [arg1, arg2, arg3] = args else {
                        return Err(MirEvalError::TypeError(
                            "fmaf32 signature doesn't match fn (f32, f32, f32) -> f32",
                        ));
                    };
                    let arg1 = from_bytes!(f32, arg1.get(self)?);
                    let arg2 = from_bytes!(f32, arg2.get(self)?);
                    let arg3 = from_bytes!(f32, arg3.get(self)?);
                    arg1.mul_add(arg2, arg3)
                }
                _ => not_supported!("unknown f32 intrinsic {name}"),
            };
            return destination.write_from_bytes(self, &result.to_le_bytes());
        }
        match name {
            "size_of" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError("size_of generic arg is not provided"));
                };
                let size = self.size_of_sized(ty, locals, "size_of arg")?;
                destination.write_from_bytes(self, &size.to_le_bytes()[0..destination.size])
            }
            "min_align_of" | "pref_align_of" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("align_of generic arg is not provided"));
                };
                let align = self.layout(ty)?.align.abi.bytes();
                destination.write_from_bytes(self, &align.to_le_bytes()[0..destination.size])
            }
            "size_of_val" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError("size_of_val generic arg is not provided"));
                };
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("size_of_val args are not provided"));
                };
                if let Some((size, _)) = self.size_align_of(ty, locals)? {
                    destination.write_from_bytes(self, &size.to_le_bytes())
                } else {
                    let metadata = arg.interval.slice(self.ptr_size()..self.ptr_size() * 2);
                    let (size, _) = self.size_align_of_unsized(ty, metadata, locals)?;
                    destination.write_from_bytes(self, &size.to_le_bytes())
                }
            }
            "min_align_of_val" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("min_align_of_val generic arg is not provided"));
                };
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("min_align_of_val args are not provided"));
                };
                if let Some((_, align)) = self.size_align_of(ty, locals)? {
                    destination.write_from_bytes(self, &align.to_le_bytes())
                } else {
                    let metadata = arg.interval.slice(self.ptr_size()..self.ptr_size() * 2);
                    let (_, align) = self.size_align_of_unsized(ty, metadata, locals)?;
                    destination.write_from_bytes(self, &align.to_le_bytes())
                }
            }
            "type_name" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError("type_name generic arg is not provided"));
                };
                let Ok(ty_name) = ty.display_source_code(
                    self.db,
                    locals.body.owner.module(self.db.upcast()),
                    true,
                ) else {
                    not_supported!("fail in generating type_name using source code display");
                };
                let len = ty_name.len();
                let addr = self.heap_allocate(len, 1)?;
                self.write_memory(addr, ty_name.as_bytes())?;
                destination.slice(0..self.ptr_size()).write_from_bytes(self, &addr.to_bytes())?;
                destination
                    .slice(self.ptr_size()..2 * self.ptr_size())
                    .write_from_bytes(self, &len.to_le_bytes())
            }
            "needs_drop" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError("size_of generic arg is not provided"));
                };
                let result = !ty.clone().is_copy(self.db, locals.body.owner);
                destination.write_from_bytes(self, &[u8::from(result)])
            }
            "ptr_guaranteed_cmp" => {
                // FIXME: this is wrong for const eval, it should return 2 in some
                // cases.
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("wrapping_add args are not provided"));
                };
                let ans = lhs.get(self)? == rhs.get(self)?;
                destination.write_from_bytes(self, &[u8::from(ans)])
            }
            "saturating_add" | "saturating_sub" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("saturating_add args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = match name {
                    "saturating_add" => lhs.saturating_add(rhs),
                    "saturating_sub" => lhs.saturating_sub(rhs),
                    _ => unreachable!(),
                };
                let bits = destination.size * 8;
                // FIXME: signed
                let is_signed = false;
                let mx: u128 = if is_signed { (1 << (bits - 1)) - 1 } else { (1 << bits) - 1 };
                // FIXME: signed
                let mn: u128 = 0;
                let ans = cmp::min(mx, cmp::max(mn, ans));
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_add" | "unchecked_add" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("wrapping_add args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_add(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_sub" | "unchecked_sub" | "ptr_offset_from_unsigned" | "ptr_offset_from" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("wrapping_sub args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_sub(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_mul" | "unchecked_mul" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("wrapping_mul args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_mul(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_shl" | "unchecked_shl" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("unchecked_shl args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_shl(rhs as u32);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "wrapping_shr" | "unchecked_shr" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("unchecked_shr args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_shr(rhs as u32);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "unchecked_rem" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("unchecked_rem args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.checked_rem(rhs).ok_or_else(|| {
                    MirEvalError::UndefinedBehavior("unchecked_rem with bad inputs".to_owned())
                })?;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "unchecked_div" | "exact_div" => {
                // FIXME: signed
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("unchecked_div args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.checked_div(rhs).ok_or_else(|| {
                    MirEvalError::UndefinedBehavior("unchecked_rem with bad inputs".to_owned())
                })?;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("const_eval_select args are not provided"));
                };
                let result_ty = TyKind::Tuple(
                    2,
                    Substitution::from_iter(Interner, [lhs.ty.clone(), TyBuilder::bool()]),
                )
                .intern(Interner);
                let op_size =
                    self.size_of_sized(&lhs.ty, locals, "operand of add_with_overflow")?;
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let (ans, u128overflow) = match name {
                    "add_with_overflow" => lhs.overflowing_add(rhs),
                    "sub_with_overflow" => lhs.overflowing_sub(rhs),
                    "mul_with_overflow" => lhs.overflowing_mul(rhs),
                    _ => unreachable!(),
                };
                let is_overflow = u128overflow
                    || ans.to_le_bytes()[op_size..].iter().any(|&it| it != 0 && it != 255);
                let is_overflow = vec![u8::from(is_overflow)];
                let layout = self.layout(&result_ty)?;
                let result = self.make_by_layout(
                    layout.size.bytes_usize(),
                    &layout,
                    None,
                    [ans.to_le_bytes()[0..op_size].to_vec(), is_overflow]
                        .into_iter()
                        .map(IntervalOrOwned::Owned),
                )?;
                destination.write_from_bytes(self, &result)
            }
            "copy" | "copy_nonoverlapping" => {
                let [src, dst, offset] = args else {
                    return Err(MirEvalError::TypeError(
                        "copy_nonoverlapping args are not provided",
                    ));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError(
                        "copy_nonoverlapping generic arg is not provided",
                    ));
                };
                let src = Address::from_bytes(src.get(self)?)?;
                let dst = Address::from_bytes(dst.get(self)?)?;
                let offset = from_bytes!(usize, offset.get(self)?);
                let size = self.size_of_sized(ty, locals, "copy_nonoverlapping ptr type")?;
                let size = offset * size;
                let src = Interval { addr: src, size };
                let dst = Interval { addr: dst, size };
                dst.write_from_interval(self, src)
            }
            "offset" | "arith_offset" => {
                let [ptr, offset] = args else {
                    return Err(MirEvalError::TypeError("offset args are not provided"));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError("offset generic arg is not provided"));
                };
                let ptr = u128::from_le_bytes(pad16(ptr.get(self)?, false));
                let offset = u128::from_le_bytes(pad16(offset.get(self)?, false));
                let size = self.size_of_sized(ty, locals, "offset ptr type")? as u128;
                let ans = ptr + offset * size;
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "assert_inhabited" | "assert_zero_valid" | "assert_uninit_valid" | "assume" => {
                // FIXME: We should actually implement these checks
                Ok(())
            }
            "forget" => {
                // We don't call any drop glue yet, so there is nothing here
                Ok(())
            }
            "transmute" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("trasmute arg is not provided"));
                };
                destination.write_from_interval(self, arg.interval)
            }
            "likely" | "unlikely" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("likely arg is not provided"));
                };
                destination.write_from_interval(self, arg.interval)
            }
            "ctpop" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("ctpop arg is not provided"));
                };
                let result = u128::from_le_bytes(pad16(arg.get(self)?, false)).count_ones();
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "ctlz" | "ctlz_nonzero" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("cttz arg is not provided"));
                };
                let result =
                    u128::from_le_bytes(pad16(arg.get(self)?, false)).leading_zeros() as usize;
                let result = result - (128 - arg.interval.size * 8);
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "cttz" | "cttz_nonzero" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("cttz arg is not provided"));
                };
                let result = u128::from_le_bytes(pad16(arg.get(self)?, false)).trailing_zeros();
                destination
                    .write_from_bytes(self, &(result as u128).to_le_bytes()[0..destination.size])
            }
            "rotate_left" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("rotate_left args are not provided"));
                };
                let lhs = &lhs.get(self)?[0..destination.size];
                let rhs = rhs.get(self)?[0] as u32;
                match destination.size {
                    1 => {
                        let r = from_bytes!(u8, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    2 => {
                        let r = from_bytes!(u16, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    4 => {
                        let r = from_bytes!(u32, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    8 => {
                        let r = from_bytes!(u64, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    16 => {
                        let r = from_bytes!(u128, lhs).rotate_left(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    s => not_supported!("destination with size {s} for rotate_left"),
                }
            }
            "rotate_right" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("rotate_right args are not provided"));
                };
                let lhs = &lhs.get(self)?[0..destination.size];
                let rhs = rhs.get(self)?[0] as u32;
                match destination.size {
                    1 => {
                        let r = from_bytes!(u8, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    2 => {
                        let r = from_bytes!(u16, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    4 => {
                        let r = from_bytes!(u32, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    8 => {
                        let r = from_bytes!(u64, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    16 => {
                        let r = from_bytes!(u128, lhs).rotate_right(rhs);
                        destination.write_from_bytes(self, &r.to_le_bytes())
                    }
                    s => not_supported!("destination with size {s} for rotate_right"),
                }
            }
            "discriminant_value" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("discriminant_value arg is not provided"));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError(
                        "discriminant_value generic arg is not provided",
                    ));
                };
                let addr = Address::from_bytes(arg.get(self)?)?;
                let size = self.size_of_sized(ty, locals, "discriminant_value ptr type")?;
                let interval = Interval { addr, size };
                let r = self.compute_discriminant(ty.clone(), interval.get(self)?)?;
                destination.write_from_bytes(self, &r.to_le_bytes()[0..destination.size])
            }
            "const_eval_select" => {
                let [tuple, const_fn, _] = args else {
                    return Err(MirEvalError::TypeError("const_eval_select args are not provided"));
                };
                let mut args = vec![const_fn.clone()];
                let TyKind::Tuple(_, fields) = tuple.ty.kind(Interner) else {
                    return Err(MirEvalError::TypeError("const_eval_select arg[0] is not a tuple"));
                };
                let layout = self.layout(&tuple.ty)?;
                for (i, field) in fields.iter(Interner).enumerate() {
                    let field = field.assert_ty_ref(Interner).clone();
                    let offset = layout.fields.offset(i).bytes_usize();
                    let addr = tuple.interval.addr.offset(offset);
                    args.push(IntervalAndTy::new(addr, field, self, locals)?);
                }
                if let Some(target) = self.db.lang_item(self.crate_id, LangItem::FnOnce) {
                    if let Some(def) = target
                        .as_trait()
                        .and_then(|it| self.db.trait_data(it).method_by_name(&name![call_once]))
                    {
                        self.exec_fn_trait(
                            def,
                            &args,
                            // FIXME: wrong for manual impls of `FnOnce`
                            Substitution::empty(Interner),
                            locals,
                            destination,
                            None,
                            span,
                        )?;
                        return Ok(());
                    }
                }
                not_supported!("FnOnce was not available for executing const_eval_select");
            }
            "read_via_copy" | "volatile_load" => {
                let [arg] = args else {
                    return Err(MirEvalError::TypeError("read_via_copy args are not provided"));
                };
                let addr = Address::from_bytes(arg.interval.get(self)?)?;
                destination.write_from_interval(self, Interval { addr, size: destination.size })
            }
            "write_bytes" => {
                let [dst, val, count] = args else {
                    return Err(MirEvalError::TypeError("write_bytes args are not provided"));
                };
                let count = from_bytes!(usize, count.get(self)?);
                let val = from_bytes!(u8, val.get(self)?);
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner))
                else {
                    return Err(MirEvalError::TypeError(
                        "write_bytes generic arg is not provided",
                    ));
                };
                let dst = Address::from_bytes(dst.get(self)?)?;
                let size = self.size_of_sized(ty, locals, "copy_nonoverlapping ptr type")?;
                let size = count * size;
                self.write_memory_using_ref(dst, size)?.fill(val);
                Ok(())
            }
            _ => not_supported!("unknown intrinsic {name}"),
        }
    }

    fn size_align_of_unsized(
        &mut self,
        ty: &Ty,
        metadata: Interval,
        locals: &Locals,
    ) -> Result<(usize, usize)> {
        Ok(match ty.kind(Interner) {
            TyKind::Str => (from_bytes!(usize, metadata.get(self)?), 1),
            TyKind::Slice(inner) => {
                let len = from_bytes!(usize, metadata.get(self)?);
                let (size, align) = self.size_align_of_sized(inner, locals, "slice inner type")?;
                (size * len, align)
            }
            TyKind::Dyn(_) => self.size_align_of_sized(
                self.vtable_map.ty_of_bytes(metadata.get(self)?)?,
                locals,
                "dyn concrete type",
            )?,
            TyKind::Adt(id, subst) => {
                let id = id.0;
                let layout = self.layout_adt(id, subst.clone())?;
                let id = match id {
                    AdtId::StructId(s) => s,
                    _ => not_supported!("unsized enum or union"),
                };
                let field_types = &self.db.field_types(id.into());
                let last_field_ty =
                    field_types.iter().rev().next().unwrap().1.clone().substitute(Interner, subst);
                let sized_part_size =
                    layout.fields.offset(field_types.iter().count() - 1).bytes_usize();
                let sized_part_align = layout.align.abi.bytes() as usize;
                let (unsized_part_size, unsized_part_align) =
                    self.size_align_of_unsized(&last_field_ty, metadata, locals)?;
                let align = sized_part_align.max(unsized_part_align) as isize;
                let size = (sized_part_size + unsized_part_size) as isize;
                // Must add any necessary padding to `size`
                // (to make it a multiple of `align`) before returning it.
                //
                // Namely, the returned size should be, in C notation:
                //
                //   `size + ((size & (align-1)) ? align : 0)`
                //
                // emulated via the semi-standard fast bit trick:
                //
                //   `(size + (align-1)) & -align`
                let size = (size + (align - 1)) & (-align);
                (size as usize, align as usize)
            }
            _ => not_supported!("unsized type other than str, slice, struct and dyn"),
        })
    }

    fn exec_atomic_intrinsic(
        &mut self,
        name: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals,
        _span: MirSpan,
    ) -> Result<()> {
        // We are a single threaded runtime with no UB checking and no optimization, so
        // we can implement these as normal functions.
        let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|it| it.ty(Interner)) else {
            return Err(MirEvalError::TypeError("atomic intrinsic generic arg is not provided"));
        };
        let Some(arg0) = args.get(0) else {
            return Err(MirEvalError::TypeError("atomic intrinsic arg0 is not provided"));
        };
        let arg0_addr = Address::from_bytes(arg0.get(self)?)?;
        let arg0_interval =
            Interval::new(arg0_addr, self.size_of_sized(ty, locals, "atomic intrinsic type arg")?);
        if name.starts_with("load_") {
            return destination.write_from_interval(self, arg0_interval);
        }
        let Some(arg1) = args.get(1) else {
            return Err(MirEvalError::TypeError("atomic intrinsic arg1 is not provided"));
        };
        if name.starts_with("store_") {
            return arg0_interval.write_from_interval(self, arg1.interval);
        }
        if name.starts_with("xchg_") {
            destination.write_from_interval(self, arg0_interval)?;
            return arg0_interval.write_from_interval(self, arg1.interval);
        }
        if name.starts_with("xadd_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs.wrapping_add(rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("xsub_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs.wrapping_sub(rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("and_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs & rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("or_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs | rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("xor_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = lhs ^ rhs;
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        if name.starts_with("nand_") {
            destination.write_from_interval(self, arg0_interval)?;
            let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
            let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
            let ans = !(lhs & rhs);
            return arg0_interval.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
        }
        let Some(arg2) = args.get(2) else {
            return Err(MirEvalError::TypeError("atomic intrinsic arg2 is not provided"));
        };
        if name.starts_with("cxchg_") || name.starts_with("cxchgweak_") {
            let dest = if arg1.get(self)? == arg0_interval.get(self)? {
                arg0_interval.write_from_interval(self, arg2.interval)?;
                (arg1.interval, true)
            } else {
                (arg0_interval, false)
            };
            let result_ty = TyKind::Tuple(
                2,
                Substitution::from_iter(Interner, [ty.clone(), TyBuilder::bool()]),
            )
            .intern(Interner);
            let layout = self.layout(&result_ty)?;
            let result = self.make_by_layout(
                layout.size.bytes_usize(),
                &layout,
                None,
                [IntervalOrOwned::Borrowed(dest.0), IntervalOrOwned::Owned(vec![u8::from(dest.1)])]
                    .into_iter(),
            )?;
            return destination.write_from_bytes(self, &result);
        }
        not_supported!("unknown atomic intrinsic {name}");
    }
}
