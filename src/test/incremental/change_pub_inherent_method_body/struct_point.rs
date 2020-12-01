// Test where we change the body of a public, inherent method.

// revisions:cfail1 cfail2
// compile-flags: -Z query-dep-graph
// build-pass

#![crate_type = "rlib"]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]

#![rustc_partition_codegened(module="struct_point-point", cfg="cfail2")]

#![rustc_partition_reused(module="struct_point-fn_calls_changed_method", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_calls_another_method", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="cfail2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="cfail2")]

pub mod point {
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    impl Point {
        pub fn distance_from_origin(&self) -> f32 {
            #[cfg(cfail1)]
            return self.x * self.x + self.y * self.y;

            #[cfg(cfail2)]
            return (self.x * self.x + self.y * self.y).sqrt();
        }

        pub fn x(&self) -> f32 {
            self.x
        }
    }
}

/// A fn item that calls the method on `Point` which changed
pub mod fn_calls_changed_method {
    use point::Point;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.distance_from_origin();
    }
}

/// A fn item that calls a method on `Point` which did not change
pub mod fn_calls_another_method {
    use point::Point;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.x();
    }
}

/// A fn item that makes an instance of `Point` but does not invoke methods
pub mod fn_make_struct {
    use point::Point;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn make_origin() -> Point {
        Point { x: 2.0, y: 2.0 }
    }
}

/// A fn item that reads fields from `Point` but does not invoke methods
pub mod fn_read_field {
    use point::Point;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A fn item that writes to a field of `Point` but does not invoke methods
pub mod fn_write_field {
    use point::Point;

    #[rustc_clean(label="typeck", cfg="cfail2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}
