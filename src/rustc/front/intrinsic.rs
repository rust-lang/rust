// NB: this file is #include_str'ed into the compiler, re-parsed
// and injected into each crate the compiler builds. Keep it small.

mod intrinsic {

    // import rusti::visit_ty;
    // import rusti::visit_val;
    // import rusti::visit_val_pair;

    export ty_visitor, val_visitor, val_pair_visitor;

    fn macros() {
        // Present for side-effect of defining intrinsic macros.
        #macro([#error[f, ...], log(core::error, #fmt[f, ...])]);
        #macro([#warn[f, ...], log(core::warn, #fmt[f, ...])]);
        #macro([#info[f, ...], log(core::info, #fmt[f, ...])]);
        #macro([#debug[f, ...], log(core::debug, #fmt[f, ...])]);
    }

    iface ty_visitor {
        fn visit_bot();
        fn visit_nil();
        fn visit_bool();

        fn visit_int();
        fn visit_i8();
        fn visit_i16();
        fn visit_i32();
        fn visit_i64();

        fn visit_uint();
        fn visit_u8();
        fn visit_u16();
        fn visit_u32();
        fn visit_u64();

        fn visit_float();
        fn visit_f32();
        fn visit_f64();

        fn visit_char();
        fn visit_str();

        fn visit_vec(cells_mut: bool,
                     visit_cell: fn(uint, self));

        fn visit_box(inner_mut: bool,
                     visit_inner: fn(self));

        fn visit_uniq(inner_mut: bool,
                      visit_inner: fn(self));

        fn visit_ptr(inner_mut: bool,
                     visit_inner: fn(self));

        fn visit_rptr(inner_mut: bool,
                      visit_inner: fn(self));

        fn visit_rec(n_fields: uint,
                     field_name: fn(uint) -> str/&,
                     field_mut: fn(uint) -> bool,
                     visit_field: fn(uint, self));
        fn visit_tup(n_fields: uint,
                     visit_field: fn(uint, self));
        fn visit_enum(n_variants: uint,
                      variant: uint,
                      variant_name: fn(uint) -> str/&,
                      visit_variant: fn(uint, self));
    }

    iface val_visitor {

        // Basic types we can visit directly.
        fn visit_bot();
        fn visit_nil();
        fn visit_bool(b: &bool);

        fn visit_int(i: &int);
        fn visit_i8(i: &i8);
        fn visit_i16(i: &i16);
        fn visit_i32(i: &i32);
        fn visit_i64(i: &i64);

        fn visit_uint(u: &uint);
        fn visit_u8(i: &i8);
        fn visit_u16(i: &i16);
        fn visit_u32(i: &i32);
        fn visit_u64(i: &i64);

        fn visit_float(f: &float);
        fn visit_f32(f: &f32);
        fn visit_f64(f: &f64);

        fn visit_char(c: &char);

        // Vecs and strs we can provide a stub view of.
        fn visit_str(repr: &vec::unsafe::vec_repr,
                     visit_cell: fn(uint,self));

        fn visit_vec(repr: &vec::unsafe::vec_repr,
                     cells_mut: bool,
                     visit_cell: fn(uint, self));

        fn visit_box(mem: *u8,
                     inner_mut: bool,
                     visit_inner: fn(self));

        fn visit_uniq(mem: *u8,
                      inner_mut: bool,
                      visit_inner: fn(self));

        fn visit_ptr(mem: *u8,
                     inner_mut: bool,
                     visit_inner: fn(self));

        fn visit_rptr(mem: *u8,
                      inner_mut: bool,
                      visit_inner: fn(self));

        // Aggregates we can't really provide anything useful for
        // beyond a *u8. You really have to know what you're doing.
        fn visit_rec(mem: *u8,
                     n_fields: uint,
                     field_name: fn(uint) -> str/&,
                     field_mut: fn(uint) -> bool,
                     visit_field: fn(uint, self));
        fn visit_tup(mem: *u8,
                     n_fields: uint,
                     visit_field: fn(uint, self));
        fn visit_enum(mem: *u8,
                      n_variants: uint,
                      variant: uint,
                      variant_name: fn(uint) -> str/&,
                      visit_variant: fn(uint, self));
    }

    iface val_pair_visitor {
    }

    #[abi = "rust-intrinsic"]
    native mod rusti {
        // fn visit_ty<T,V:ty_visitor>(tv: V);
        // fn visit_val<T,V:val_visitor>(v: &T, vv: V);
        // fn visit_val_pair<T,V:val_pair_visitor>(a: &T, b: &T, vpv: &V);
    }
}
