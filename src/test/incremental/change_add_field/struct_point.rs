// Test where we change a type definition by adding a field.  Fns with
// this type in their signature are recompiled, as are their callers.
// Fns with that type used only in their body are also recompiled, but
// their callers are not.

// revisions:cfail1 cfail2
// compile-flags: -Z query-dep-graph
// build-pass

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
#![crate_type = "rlib"]

// These are expected to require codegen.
#![rustc_partition_codegened(module="struct_point-point", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-fn_with_type_in_sig", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-call_fn_with_type_in_sig", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-fn_with_type_in_body", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-fn_make_struct", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-fn_read_field", cfg="cfail2")]
#![rustc_partition_codegened(module="struct_point-fn_write_field", cfg="cfail2")]

#![rustc_partition_reused(module="struct_point-call_fn_with_type_in_body", cfg="cfail2")]

pub mod point {
    #[cfg(cfail1)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    #[cfg(cfail2)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    impl Point {
        pub fn origin() -> Point {
            #[cfg(cfail1)]
            return Point { x: 0.0, y: 0.0 };

            #[cfg(cfail2)]
            return Point { x: 0.0, y: 0.0, z: 0.0 };
        }

        pub fn total(&self) -> f32 {
            #[cfg(cfail1)]
            return self.x + self.y;

            #[cfg(cfail2)]
            return self.x + self.y + self.z;
        }

        pub fn x(&self) -> f32 {
            self.x
        }
    }
}

/// A function that has the changed type in its signature; must currently be
/// rebuilt.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
pub mod fn_with_type_in_sig {
    use point::Point;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn boop(p: Option<&Point>) -> f32 {
        p.map(|p| p.total()).unwrap_or(0.0)
    }
}

/// Call a function that has the changed type in its signature; this
/// currently must also be rebuilt.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
pub mod call_fn_with_type_in_sig {
    use fn_with_type_in_sig;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn bip() -> f32 {
        fn_with_type_in_sig::boop(None)
    }
}

/// A function that uses the changed type, but only in its body, not its
/// signature.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
pub mod fn_with_type_in_body {
    use point::Point;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn boop() -> f32 {
        Point::origin().total()
    }
}

/// A function `X` that calls a function `Y`, where `Y` uses the changed type in its
/// body. In this case, the effects of the change should be contained
/// to `Y`; `X` should not have to be rebuilt, nor should it need to be
/// type-checked again.
pub mod call_fn_with_type_in_body {
    use fn_with_type_in_body;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn bip() -> f32 {
        fn_with_type_in_body::boop()
    }
}

/// A function item that makes an instance of `Point` but does not invoke methods.
pub mod fn_make_struct {
    use point::Point;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn make_origin(p: Point) -> Point {
        Point { ..p }
    }
}

/// A function item that reads fields from `Point` but does not invoke methods.
pub mod fn_read_field {
    use point::Point;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A function item that writes to a field of `Point` but does not invoke methods.
pub mod fn_write_field {
    use point::Point;

    #[rustc_dirty(label="typeck", cfg="cfail2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}
