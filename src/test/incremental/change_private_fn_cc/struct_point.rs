// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test where we change the body of a private method in an impl.
// We then test what sort of functions must be rebuilt as a result.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph
// aux-build:point.rs

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![allow(dead_code)]

#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_same_impl", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_calls_methods_in_another_impl", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_read_field", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_write_field", cfg="rpass2")]
#![rustc_partition_reused(module="struct_point-fn_make_struct", cfg="rpass2")]

extern crate point;

/// A fn item that calls (public) methods on `Point` from the same impl which changed
mod fn_calls_methods_in_same_impl {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn check() {
        let x = Point { x: 2.0, y: 2.0 };
        x.distance_from_origin();
    }
}

/// A fn item that calls (public) methods on `Point` from another impl
mod fn_calls_methods_in_another_impl {
    use point::Point;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn check() {
        let mut x = Point { x: 2.0, y: 2.0 };
        x.translate(3.0, 3.0);
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
