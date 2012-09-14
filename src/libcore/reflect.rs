use intrinsic::{TyDesc, get_tydesc, visit_tydesc, TyVisitor};
use libc::c_void;

export visit_data, DataVisitor;

trait MovablePtr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void);
}

// FIXME #3262: this is a near-duplicate of code in core::vec.
type VecRepr = {
    box_header: (uint, uint, uint, uint),
    mut fill: uint,
    mut alloc: uint,
    data: u8
};
type SliceRepr = {
    mut data: *u8,
    mut len: uint
};

/// Helper function for alignment calculation.
#[inline(always)]
fn align(size: uint, align: uint) -> uint {
    ((size + align) - 1u) & !(align - 1u)
}

// Helper function for vec visiting
fn visit_vec_helper<T: DataVisitor Owned>(visitor: DataVisitorAdaptor<T>,
                    inner: *TyDesc, start_marker: fn(uint),
                    elem_marker: fn()) {
    visitor.align_to::<*VecRepr>();
    do visitor.get::<*VecRepr> |v| {
        do visitor.jump(v) {
            // skip the header
            visitor.bump_past::<(uint, uint, uint, uint)>();
            do visitor.get::<uint> |fill| {
                let num;
                unsafe {
                    num = fill / (*inner).size;
                }
                start_marker(num);
                // skip the fill and alloc
                visitor.bump_past::<(uint, uint)>();
                for num.times {
                    elem_marker();
                    visitor.visit_rec(inner);
                }
            }
        }
        visitor.bump_past::<*VecRepr>();
    }
}

enum DataVisitorAdaptor<V: DataVisitor Owned> = @{
    mut ptr: *c_void,
    // for certain things, like enum variant, we want to ignore visited types
    mut ignore: bool,
    inner: V
};

impl<V: DataVisitor Owned> DataVisitorAdaptor<V> {
    fn get<T>(f: fn(T)) {
        unsafe {
            f(*(self.ptr as *T));
        }
    }

    fn visit_rec(ty: *TyDesc) -> bool {
        let u = DataVisitorAdaptor(*self);
        visit_tydesc(ty, u as TyVisitor);
        true
    }

    fn jump<T>(p: *T, blk: fn()) {
        let cur = self.ptr;
        self.move_ptr(|_p| { p as *c_void });
        blk();
        self.move_ptr(|_p| { cur });
    }
}

impl<V: DataVisitor Owned> DataVisitorAdaptor<V>: MovablePtr {
    fn move_ptr(adjustment: fn(*c_void) -> *c_void) {
        self.ptr = adjustment(self.ptr);
    }
}


trait DataVisitor {
    fn visit_nil();
    fn visit_bool(val: bool);
    fn visit_int(val: int);
    fn visit_i8(val: i8);
    fn visit_i16(val: i16);
    fn visit_i32(val: i32);
    fn visit_i64(val: i64);
    fn visit_uint(val: uint);
    fn visit_u8(val: u8);
    fn visit_u16(val: u16);
    fn visit_u32(val: u32);
    fn visit_u64(val: u64);
    fn visit_float(val: float);
    fn visit_f32(val: f32);
    fn visit_f64(val: f32);
    fn visit_char(val: char);
    // is this type used?
    // fn visit_str(val: str)
    fn visit_str_box(val: @str);
    fn visit_str_uniq(val: ~str);
    fn visit_str_slice(val: &str);
    fn visit_str_fixed(val: &str); // can't actually pass fixed size
    fn visit_box(); // indicates next visit is in a box
    fn visit_uniq();
    //fn visit_ptr // not sure what these are
    //fn visit_rptr
    // is this type used?
    // fn visit_vec(val: vec)
    fn visit_vec_box_start(size: uint);
    fn visit_vec_box_elem();
    fn visit_vec_box_end();
    fn visit_vec_uniq_start(size: uint);
    fn visit_vec_uniq_elem();
    fn visit_vec_uniq_end();
    fn visit_vec_slice_start(size: uint);
    fn visit_vec_slice_elem();
    fn visit_vec_slice_end();
    fn visit_vec_fixed_start(size: uint);
    fn visit_vec_fixed_elem();
    fn visit_vec_fixed_end();
    fn visit_rec_start(fields: uint);
    fn visit_rec_field(name: &str); // next visit is value
    fn visit_rec_end();
    // what is a class now? are the visit_class_* all deprecated
    fn visit_tup_start(size: uint);
    fn visit_tup_elem();
    fn visit_tup_end();
    fn visit_enum_start(size: uint, disr_val: int, name: &str);
    fn visit_enum_field(); // next is value
    fn visit_enum_field_end();
    // Still needed:
    // functions, closures, traits, paraws, self
}


impl<V: DataVisitor Owned> DataVisitorAdaptor<V>: TyVisitor {
  #[inline(always)]
    fn bump(sz: uint) {
      do self.move_ptr |p| {
            ((p as uint) + sz) as *c_void
      };
    }

    #[inline(always)]
    fn align(a: uint) {
      do self.move_ptr |p| {
            align(p as uint, a) as *c_void
      };
    }

    #[inline(always)]
    fn align_to<T>() {
        self.align(sys::min_align_of::<T>());
    }

    #[inline(always)]
    fn bump_past<T>() {
        self.bump(sys::size_of::<T>());
    }

    fn visit_bot() -> bool { true }
    fn visit_nil() -> bool {
        self.inner.visit_nil();
        true
    }
    fn visit_bool() -> bool {
        self.align_to::<bool>();
        do self.get::<bool> |b| {
            self.inner.visit_bool(b);
        };
        self.bump_past::<bool>();
        true
    }
    fn visit_int() -> bool {
        self.align_to::<int>();
        do self.get::<int> |i| {
            self.inner.visit_int(i);
        };
        self.bump_past::<int>();
        true
    }
    fn visit_i8() -> bool {
        self.align_to::<i8>();
        do self.get::<i8> |i| {
            self.inner.visit_i8(i);
        };
        self.bump_past::<i8>();
        true
    }
    fn visit_i16() -> bool {
        self.align_to::<i16>();
        do self.get::<i16> |i| {
            self.inner.visit_i16(i);
        };
        self.bump_past::<i16>();
        true
    }
    fn visit_i32() -> bool {
        self.align_to::<i32>();
        do self.get::<i32> |i| {
            self.inner.visit_i32(i);
        };
        self.bump_past::<i32>();
        true
    }
    fn visit_i64() -> bool {
        self.align_to::<i64>();
        do self.get::<i64> |i| {
            self.inner.visit_i64(i);
        };
        self.bump_past::<i64>();
        true
    }

    fn visit_uint() -> bool {
        self.align_to::<uint>();
        do self.get::<uint> |i| {
            self.inner.visit_uint(i);
        };
        self.bump_past::<uint>();
        true
    }
    fn visit_u8() -> bool {
        self.align_to::<u8>();
        do self.get::<u8> |i| {
            self.inner.visit_u8(i);
        };
        self.bump_past::<u8>();
        true
    }
    fn visit_u16() -> bool {
        self.align_to::<u16>();
        do self.get::<u16> |i| {
            self.inner.visit_u16(i);
        };
        self.bump_past::<u16>();
        true
    }
    fn visit_u32() -> bool {
        self.align_to::<u32>();
        do self.get::<u32> |i| {
            self.inner.visit_u32(i);
        };
        self.bump_past::<u32>();
        true
    }
    fn visit_u64() -> bool {
        self.align_to::<u64>();
        do self.get::<u64> |i| {
            self.inner.visit_u64(i);
        };
        self.bump_past::<u64>();
        true
    }

    fn visit_float() -> bool {
        self.align_to::<float>();
        do self.get::<float> |i| {
            self.inner.visit_float(i);
        };
        self.bump_past::<float>();
        true
    }
    fn visit_f32() -> bool {
        // self.align_to::<f32>();
        // do self.get::<f32>() |i| {
        //     self.out += f32::to_str(i, 10u);
        // };
        // self.bump_past::<f32>();
        true
    }
    fn visit_f64() -> bool {
        // self.align_to::<f64>();
        // do self.get::<f64>() |i| {
        //     self.out += f64::to_str(i, 10u);
        // };
        // self.bump_past::<f64>();
        true
    }

    fn visit_char() -> bool {
        self.align_to::<char>();
        do self.get::<char> |c| {
            self.inner.visit_char(c);
        };
        self.bump_past::<char>();
        true
    }
    // is this fn used?
    fn visit_str() -> bool {
        // self.align_to::<~str>();
        // self.out += ~"\"";

        // do self.get::<~str> |s| {
        // };
        // self.bump_past::<~str>();
        // self.out += ~"\"";
        true
    }

    fn visit_estr_box() -> bool {
        self.align_to::<@str>();
        do self.get::<@str> |s| {
            self.inner.visit_str_box(s);
        };
        self.bump_past::<@str>();
        true
    }
    fn visit_estr_uniq() -> bool {
        self.align_to::<~str>();
        do self.get::<~str> |s| {
            self.inner.visit_str_uniq(s);
        };
        self.bump_past::<~str>();
        true
    }
    fn visit_estr_slice() -> bool {
        self.align_to::<&static/str>();
        do self.get::<&static/str> |s| {
            self.inner.visit_str_slice(s);
        };
        self.bump_past::<&static/str>();
        true
    }
    fn visit_estr_fixed(_n: uint, _sz: uint,
                        _align: uint) -> bool {
        self.align(_align);
        unsafe {
            self.inner.visit_str_fixed(vec::raw::form_slice(
                self.ptr as *u8, _n, str::from_bytes));
        }
        self.bump(_sz);
        true
    }

    fn visit_box(_mtbl: uint, _inner: *TyDesc) -> bool {
        // Question - it looks like fmt doesn't distinguish between
        // mutable and not - that true?
        self.inner.visit_box();
        self.visit_rec(_inner);
        true
    }
    fn visit_uniq(_mtbl: uint, _inner: *TyDesc) -> bool {
        self.inner.visit_uniq();
        self.visit_rec(_inner);
        true
    }
    // Not sure what these are for
    fn visit_ptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_rptr(_mtbl: uint, _inner: *TyDesc) -> bool { true }
    fn visit_unboxed_vec(_mtbl: uint, _inner: *TyDesc) -> bool {
        true
    }
    fn visit_vec(_mtbl: uint, _inner: *TyDesc) -> bool {
        // Is this used?
        true
    }
    fn visit_evec_box(_mtbl: uint, _inner: *TyDesc) -> bool {
        visit_vec_helper(self, _inner,
            |size| { self.inner.visit_vec_box_start(size); },
            || { self.inner.visit_vec_box_elem() });
        self.inner.visit_vec_box_end();
        true
    }
    fn visit_evec_uniq(_mtbl: uint, _inner: *TyDesc) -> bool {
        visit_vec_helper(self, _inner,
            |size| { self.inner.visit_vec_uniq_start(size); },
            || { self.inner.visit_vec_uniq_elem() });
        self.inner.visit_vec_uniq_end();
        true
    }
    fn visit_evec_slice(_mtbl: uint, _inner: *TyDesc) -> bool {
        do self.get::<SliceRepr> |d| {
            let num;
            unsafe {
                num = d.len / (*_inner).size;
            }
            self.inner.visit_vec_slice_start(num);
            do self.jump(d.data) {
                for num.times {
                    self.inner.visit_vec_slice_elem();
                    self.visit_rec(_inner);
                }
            }
        }
        self.bump_past::<SliceRepr>();
        self.inner.visit_vec_slice_end();
        true
    }
    fn visit_evec_fixed(_n: uint, _sz: uint, _align: uint,
                        _mtbl: uint, _inner: *TyDesc) -> bool {
        self.align(_align);
        self.inner.visit_vec_fixed_start(_n);
        for _n.times {
            self.inner.visit_vec_fixed_elem();
            self.visit_rec(_inner);
        }
        self.inner.visit_vec_fixed_end();
        true
    }

    fn visit_enter_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.align(_align);
        self.inner.visit_rec_start(_n_fields);
        true
    }
    fn visit_rec_field(_i: uint, _name: &str,
                       _mtbl: uint, inner: *TyDesc) -> bool {
        self.inner.visit_rec_field(_name);
        self.visit_rec(inner);
        true
    }
    fn visit_leave_rec(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.inner.visit_rec_end();
        true
    }

    fn visit_enter_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }
    fn visit_class_field(_i: uint, _name: &str,
                         _mtbl: uint, inner: *TyDesc) -> bool {
        self.visit_rec(inner)
    }
    fn visit_leave_class(_n_fields: uint,
                         _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.align(_align);
        self.inner.visit_tup_start(_n_fields);
        true
    }
    fn visit_tup_field(_i: uint, inner: *TyDesc) -> bool {
        self.inner.visit_tup_elem();
        self.visit_rec(inner);
        true
    }
    fn visit_leave_tup(_n_fields: uint,
                       _sz: uint, _align: uint) -> bool {
        self.inner.visit_tup_end();
        true
    }

    fn visit_enter_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool {
        self.align(_align);
        true
    }
    fn visit_enter_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool {
        do self.get::<int> |e| {
            if e == _disr_val {
                self.inner.visit_enum_start(_n_fields, _disr_val, _name);
                self.bump_past::<int>();
            } else {
                self.ignore = true;
            }
        };
        true
    }
    fn visit_enum_variant_field(_i: uint, inner: *TyDesc) -> bool {
        if !self.ignore {
            self.inner.visit_enum_field();
            self.visit_rec(inner)
        } else {
            true
        }
    }
    fn visit_leave_enum_variant(_variant: uint,
                                _disr_val: int,
                                _n_fields: uint,
                                _name: &str) -> bool {
        if !self.ignore {
            self.inner.visit_enum_field_end();
        } else {
            // no matter whether we were ignoring or not, don't anymore!
            self.ignore = false;
        }
        true
    }
    fn visit_leave_enum(_n_variants: uint,
                        _sz: uint, _align: uint) -> bool { true }

    fn visit_enter_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }
    fn visit_fn_input(_i: uint, _mode: uint, _inner: *TyDesc) -> bool { true }
    fn visit_fn_output(_retstyle: uint, _inner: *TyDesc) -> bool { true }
    fn visit_leave_fn(_purity: uint, _proto: uint,
                      _n_inputs: uint, _retstyle: uint) -> bool { true }


    fn visit_trait() -> bool { true }
    fn visit_var() -> bool { true }
    fn visit_var_integral() -> bool { true }
    fn visit_param(_i: uint) -> bool { true }
    fn visit_self() -> bool { true }
    fn visit_type() -> bool { true }
    fn visit_opaque_box() -> bool { true }
    fn visit_constr(_inner: *TyDesc) -> bool { true }
    fn visit_closure_ptr(_ck: uint) -> bool { true }
}

fn get_tydesc_for<T>(&&_t: T) -> *TyDesc {
    get_tydesc::<T>()
}

fn visit_data<D: DataVisitor Copy Owned, T>(&visitor: D, value: T) {
    let p = ptr::addr_of(value) as *c_void;
    let v = DataVisitorAdaptor(@{mut ptr: p,
                                   mut ignore: false,
                                   inner: visitor});
    unsafe {
        let td = get_tydesc_for(value);
        visit_tydesc(td, v as TyVisitor);
    }
}