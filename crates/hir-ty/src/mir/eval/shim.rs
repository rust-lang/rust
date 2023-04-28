//! Interpret intrinsics, lang items and `extern "C"` wellknown functions which their implementation
//! is not available.

use super::*;

macro_rules! from_bytes {
    ($ty:tt, $value:expr) => {
        ($ty::from_le_bytes(match ($value).try_into() {
            Ok(x) => x,
            Err(_) => return Err(MirEvalError::TypeError("mismatched size")),
        }))
    };
}

macro_rules! not_supported {
    ($x: expr) => {
        return Err(MirEvalError::NotSupported(format!($x)))
    };
}

impl Evaluator<'_> {
    pub(super) fn detect_and_exec_special_function(
        &mut self,
        def: FunctionId,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        locals: &Locals<'_>,
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
        let alloc_fn = function_data
            .attrs
            .iter()
            .filter_map(|x| x.path().as_ident())
            .filter_map(|x| x.as_str())
            .find(|x| {
                [
                    "rustc_allocator",
                    "rustc_deallocator",
                    "rustc_reallocator",
                    "rustc_allocator_zeroed",
                ]
                .contains(x)
            });
        if let Some(alloc_fn) = alloc_fn {
            self.exec_alloc_fn(alloc_fn, args, destination)?;
            return Ok(true);
        }
        if let Some(x) = self.detect_lang_function(def) {
            let arg_bytes =
                args.iter().map(|x| Ok(x.get(&self)?.to_owned())).collect::<Result<Vec<_>>>()?;
            let result = self.exec_lang_item(x, &arg_bytes)?;
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
                let result = self.heap_allocate(size, align);
                destination.write_from_bytes(self, &result.to_bytes())?;
            }
            "rustc_deallocator" => { /* no-op for now */ }
            "rustc_reallocator" => {
                let [ptr, old_size, align, new_size] = args else {
                    return Err(MirEvalError::TypeError("rustc_allocator args are not provided"));
                };
                let ptr = Address::from_bytes(ptr.get(self)?)?;
                let old_size = from_bytes!(usize, old_size.get(self)?);
                let new_size = from_bytes!(usize, new_size.get(self)?);
                let align = from_bytes!(usize, align.get(self)?);
                let result = self.heap_allocate(new_size, align);
                Interval { addr: result, size: old_size }
                    .write_from_interval(self, Interval { addr: ptr, size: old_size })?;
                destination.write_from_bytes(self, &result.to_bytes())?;
            }
            _ => not_supported!("unknown alloc function"),
        }
        Ok(())
    }

    fn detect_lang_function(&self, def: FunctionId) -> Option<LangItem> {
        use LangItem::*;
        let candidate = lang_attr(self.db.upcast(), def)?;
        // We want to execute these functions with special logic
        if [PanicFmt, BeginPanic, SliceLen].contains(&candidate) {
            return Some(candidate);
        }
        None
    }

    fn exec_lang_item(&self, x: LangItem, args: &[Vec<u8>]) -> Result<Vec<u8>> {
        use LangItem::*;
        let mut args = args.iter();
        match x {
            // FIXME: we want to find the panic message from arguments, but it wouldn't work
            // currently even if we do that, since macro expansion of panic related macros
            // is dummy.
            PanicFmt | BeginPanic => Err(MirEvalError::Panic("<format-args>".to_string())),
            SliceLen => {
                let arg = args
                    .next()
                    .ok_or(MirEvalError::TypeError("argument of <[T]>::len() is not provided"))?;
                let ptr_size = arg.len() / 2;
                Ok(arg[ptr_size..].into())
            }
            x => not_supported!("Executing lang item {x:?}"),
        }
    }

    fn exec_intrinsic(
        &mut self,
        as_str: &str,
        args: &[IntervalAndTy],
        generic_args: &Substitution,
        destination: Interval,
        locals: &Locals<'_>,
        span: MirSpan,
    ) -> Result<()> {
        // We are a single threaded runtime with no UB checking and no optimization, so
        // we can implement these as normal functions.
        if let Some(name) = as_str.strip_prefix("atomic_") {
            let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                return Err(MirEvalError::TypeError("atomic intrinsic generic arg is not provided"));
            };
            let Some(arg0) = args.get(0) else {
                return Err(MirEvalError::TypeError("atomic intrinsic arg0 is not provided"));
            };
            let arg0_addr = Address::from_bytes(arg0.get(self)?)?;
            let arg0_interval = Interval::new(
                arg0_addr,
                self.size_of_sized(ty, locals, "atomic intrinsic type arg")?,
            );
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
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
            }
            if name.starts_with("xsub_") {
                destination.write_from_interval(self, arg0_interval)?;
                let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
                let ans = lhs.wrapping_sub(rhs);
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
            }
            if name.starts_with("and_") {
                destination.write_from_interval(self, arg0_interval)?;
                let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
                let ans = lhs & rhs;
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
            }
            if name.starts_with("or_") {
                destination.write_from_interval(self, arg0_interval)?;
                let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
                let ans = lhs | rhs;
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
            }
            if name.starts_with("xor_") {
                destination.write_from_interval(self, arg0_interval)?;
                let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
                let ans = lhs ^ rhs;
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
            }
            if name.starts_with("nand_") {
                destination.write_from_interval(self, arg0_interval)?;
                let lhs = u128::from_le_bytes(pad16(arg0_interval.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(arg1.get(self)?, false));
                let ans = !(lhs & rhs);
                return arg0_interval
                    .write_from_bytes(self, &ans.to_le_bytes()[0..destination.size]);
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
                    [
                        IntervalOrOwned::Borrowed(dest.0),
                        IntervalOrOwned::Owned(vec![u8::from(dest.1)]),
                    ]
                    .into_iter(),
                )?;
                return destination.write_from_bytes(self, &result);
            }
            not_supported!("unknown atomic intrinsic {name}");
        }
        match as_str {
            "size_of" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("size_of generic arg is not provided"));
                };
                let size = self.size_of_sized(ty, locals, "size_of arg")?;
                destination.write_from_bytes(self, &size.to_le_bytes()[0..destination.size])
            }
            "min_align_of" | "pref_align_of" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("align_of generic arg is not provided"));
                };
                let align = self.layout_filled(ty, locals)?.align.abi.bytes();
                destination.write_from_bytes(self, &align.to_le_bytes()[0..destination.size])
            }
            "needs_drop" => {
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
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
            "wrapping_add" => {
                let [lhs, rhs] = args else {
                    return Err(MirEvalError::TypeError("wrapping_add args are not provided"));
                };
                let lhs = u128::from_le_bytes(pad16(lhs.get(self)?, false));
                let rhs = u128::from_le_bytes(pad16(rhs.get(self)?, false));
                let ans = lhs.wrapping_add(rhs);
                destination.write_from_bytes(self, &ans.to_le_bytes()[0..destination.size])
            }
            "add_with_overflow" => {
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
                let ans = lhs.wrapping_add(rhs);
                let is_overflow = false;
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
                    return Err(MirEvalError::TypeError("copy_nonoverlapping args are not provided"));
                };
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
                    return Err(MirEvalError::TypeError("copy_nonoverlapping generic arg is not provided"));
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
                let Some(ty) = generic_args.as_slice(Interner).get(0).and_then(|x| x.ty(Interner)) else {
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
                self.exec_fn_trait(&args, destination, locals, span)
            }
            _ => not_supported!("unknown intrinsic {as_str}"),
        }
    }
}
