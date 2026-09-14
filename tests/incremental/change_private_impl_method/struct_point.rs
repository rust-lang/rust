// Test where we change the body of a private method in an impl.
// We then test what sort of functions must be rebuilt as a result.

//@ revisions: bpass1 bpass2
//@ compile-flags: -Z query-dep-graph
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![allow(dead_code)]
#![crate_type = "rlib"]

#![rustc_partition_codegened(module="struct_point-point", cfg="bpass2")]

#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_same_impl", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_another_impl", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="bpass2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="bpass2")]

pub mod point {
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    impl Point {
        pub fn distance_squared(&self) -> f32 {
            #[cfg(bpass1)]
            return self.x + self.y;

            #[cfg(bpass2)]
            return self.x * self.x + self.y * self.y;
        }

        pub fn distance_from_origin(&self) -> f32 {
            self.distance_squared().sqrt()
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

    #[rustc_clean(cfg="bpass2")]
    pub fn check() {
        let x = Point { x: 2.0, y: 2.0 };
        x.distance_from_origin();
    }
}

/// A fn item that calls (public) methods on `Point` from another impl
pub mod fn_calls_methods_in_another_impl {
    use point::Point;

    #[rustc_clean(cfg="bpass2")]
    pub fn check() {
        let mut x = Point { x: 2.0, y: 2.0 };
        x.translate(3.0, 3.0);
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
