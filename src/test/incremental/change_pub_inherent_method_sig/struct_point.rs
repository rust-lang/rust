// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test where we change the *signature* of a public, inherent method.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(static_in_const)]
#![allow(dead_code)]

// These are expected to require translation.
#![rustc_partition_translated(module="struct_point-point", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_calls_changed_method", cfg="rpass2")]

#![rustc_partition_reused(module="struct_point-fn_calls_another_method", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="rpass2")]

mod point {
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    impl Point {
        #[cfg(rpass1)]
        pub fn distance_from_point(&self, p: Option<Point>) -> f32 {
            let p = p.unwrap_or(Point { x: 0.0, y: 0.0 });
            let x_diff = self.x - p.x;
            let y_diff = self.y - p.y;
            return x_diff * x_diff + y_diff * y_diff;
        }

        #[cfg(rpass2)]
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
mod fn_calls_changed_method {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.distance_from_point(None);
    }
}

/// A fn item that calls a method that was not changed
mod fn_calls_another_method {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn check() {
        let p = Point { x: 2.0, y: 2.0 };
        p.x();
    }
}

/// A fn item that makes an instance of `Point` but does not invoke methods
mod fn_make_struct {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn make_origin() -> Point {
        Point { x: 2.0, y: 2.0 }
    }
}

/// A fn item that reads fields from `Point` but does not invoke methods
mod fn_read_field {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A fn item that writes to a field of `Point` but does not invoke methods
mod fn_write_field {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}

fn main() {
}
