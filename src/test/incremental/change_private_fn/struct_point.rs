// Test where we change the body of a private method in an impl.
// We then test what sort of functions must be rebuilt as a result.

// revisions:cfail1 cfail2
// compile-flags: -Z query-dep-graph
// build-pass (FIXME(62277): could be check-pass?)

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="struct_point-point", cfg="cfail2")]

#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_same_impl", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_another_impl", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="cfail2")]

pub mod point {
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    fn distance_squared(this: &Point) -> f32 {
        #[cfg(cfail1)]
        return this.x + this.y;

        #[cfg(cfail2)]
        return this.x * this.x + this.y * this.y;
    }

    impl Point {
        pub fn distance_from_origin(&self) -> f32 {
            distance_squared(self).sqrt()
        }
    }

    impl Point {
        pub fn translate(&mut self, x: f32, y: f32) {
            self.x += x;
            self.y += y;
        }
    }

}

/// A fn item that calls (public) methods on `Point` from the same impl which changed
pub mod fn_calls_methods_in_same_impl {
    use point::Point;

    // The cached result should actually be loaded from disk
    // (not just marked green) - for example, `DeadVisitor`
    // always runs during compilation as a "pass", and loads
    // the typeck results for bodies.
    #[rustc_clean(cfg="cfail2", loaded_from_disk="typeck")]
    pub fn check() {
        let x = Point { x: 2.0, y: 2.0 };
        x.distance_from_origin();
    }
}

/// A fn item that calls (public) methods on `Point` from another impl
pub mod fn_calls_methods_in_another_impl {
    use point::Point;

    #[rustc_clean(cfg="cfail2")]
    pub fn check() {
        let mut x = Point { x: 2.0, y: 2.0 };
        x.translate(3.0, 3.0);
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
