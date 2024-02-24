// Test where we add a private item into the root of an external.
// crate. This should not cause anything we use to be invalidated.
// Regression test for #36168.

//@ revisions:cfail1 cfail2
//@ compile-flags: -Z query-dep-graph
//@ aux-build:point.rs
//@ build-pass

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
#![crate_type = "rlib"]

#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_same_impl", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_calls_free_fn", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="cfail2")]

extern crate point;

/// A fn item that calls (public) methods on `Point` from the same impl
pub mod fn_calls_methods_in_same_impl {
    use point::Point;

    #[rustc_clean(cfg="cfail2")]
    pub fn check() {
        let x = Point { x: 2.0, y: 2.0 };
        x.distance_from_origin();
    }
}

/// A fn item that calls (public) methods on `Point` from another impl
pub mod fn_calls_free_fn {
    use point::{self, Point};

    #[rustc_clean(cfg="cfail2")]
    pub fn check() {
        let x = Point { x: 2.0, y: 2.0 };
        point::distance_squared(&x);
    }
}

/// A fn item that makes an instance of `Point` but does not invoke methods
pub mod fn_make_struct {
    use point::Point;

    #[rustc_clean(cfg="cfail2")]
    pub fn make_origin() -> Point {
        Point { x: 2.0, y: 2.0 }
    }
}

/// A fn item that reads fields from `Point` but does not invoke methods
pub mod fn_read_field {
    use point::Point;

    #[rustc_clean(cfg="cfail2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A fn item that writes to a field of `Point` but does not invoke methods
pub mod fn_write_field {
    use point::Point;

    #[rustc_clean(cfg="cfail2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}
