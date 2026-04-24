// Test where we change the *signature* of a public, inherent method.

//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![crate_type = "rlib"]
#![feature(rustc_attrs)]
#![allow(dead_code)]

// These are expected to require codegen.
#![rustc_partition_codegened(module="struct_point-point", cfg="bpass2")]
#![rustc_partition_codegened(module="struct_point-fn_calls_changed_method", cfg="bpass2")]

#![rustc_partition_reused(module="struct_point-fn_calls_another_method", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="bpass2")]

pub mod point {
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    impl Point {
        #[cfg(bpass1)]
        pub fn distance_from_point(&self, p: Option<Point>) -> f32 {
            let p = p.unwrap_or(Point { x: 0.0, y: 0.0 });
            let x_diff = self.x - p.x;
            let y_diff = self.y - p.y;
            return x_diff * x_diff + y_diff * y_diff;
        }

        #[cfg(bpass2)]
        pub fn distance_from_point(&self, p: Option<&Point>) -> f32 {
            const ORIGIN: &Point = &Point { x: 0.0, y: 0.0 };
            let p = p.unwrap_or(ORIGIN);
            let x_diff = self.x - p.x;
            let y_diff = self.y - p.y;
            return x_diff * x_diff + y_diff * y_diff;
        }

        pub fn x(&self) -> f32 {
            self.x
        }
    }
}

/// A fn item that calls the method that was changed
pub mod fn_calls_changed_method {
    use point::Point;

    #[rustc_clean(except="typeck_root,optimized_mir", cfg="bpass2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.distance_from_point(None);
    }
}

/// A fn item that calls a method that was not changed
pub mod fn_calls_another_method {
    use point::Point;

    #[rustc_clean(cfg="bpass2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.x();
    }
}

/// A fn item that makes an instance of `Point` but does not invoke methods
pub mod fn_make_struct {
    use point::Point;

    #[rustc_clean(cfg="bpass2")]
    pub fn make_origin() -> Point {
        Point { x: 2.0, y: 2.0 }
    }
}

/// A fn item that reads fields from `Point` but does not invoke methods
pub mod fn_read_field {
    use point::Point;

    #[rustc_clean(cfg="bpass2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A fn item that writes to a field of `Point` but does not invoke methods
pub mod fn_write_field {
    use point::Point;

    #[rustc_clean(cfg="bpass2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}
