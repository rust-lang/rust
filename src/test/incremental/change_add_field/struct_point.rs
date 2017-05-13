// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test where we change a type definition by adding a field.  Fns with
// this type in their signature are recompiled, as are their callers.
// Fns with that type used only in their body are also recompiled, but
// their callers are not.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![feature(static_in_const)]
#![allow(dead_code)]

// These are expected to require translation.
#![rustc_partition_translated(module="struct_point-point", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_with_type_in_sig", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-call_fn_with_type_in_sig", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_with_type_in_body", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_make_struct", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_read_field", cfg="rpass2")]
#![rustc_partition_translated(module="struct_point-fn_write_field", cfg="rpass2")]

#![rustc_partition_reused(module="struct_point-call_fn_with_type_in_body", cfg="rpass2")]

mod point {
    #[cfg(rpass1)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
    }

    #[cfg(rpass2)]
    pub struct Point {
        pub x: f32,
        pub y: f32,
        pub z: f32,
    }

    impl Point {
        pub fn origin() -> Point {
            #[cfg(rpass1)]
            return Point { x: 0.0, y: 0.0 };

            #[cfg(rpass2)]
            return Point { x: 0.0, y: 0.0, z: 0.0 };
        }

        pub fn total(&self) -> f32 {
            #[cfg(rpass1)]
            return self.x + self.y;

            #[cfg(rpass2)]
            return self.x + self.y + self.z;
        }

        pub fn x(&self) -> f32 {
            self.x
        }
    }
}

/// A fn that has the changed type in its signature; must currently be
/// rebuilt.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
mod fn_with_type_in_sig {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn boop(p: Option<&Point>) -> f32 {
        p.map(|p| p.total()).unwrap_or(0.0)
    }
}

/// Call a fn that has the changed type in its signature; this
/// currently must also be rebuilt.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
mod call_fn_with_type_in_sig {
    use fn_with_type_in_sig;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn bip() -> f32 {
        fn_with_type_in_sig::boop(None)
    }
}

/// A fn that uses the changed type, but only in its body, not its
/// signature.
///
/// You could imagine that, in the future, if the change were
/// sufficiently "private", we might not need to type-check again.
/// Rebuilding is probably always necessary since the layout may be
/// affected.
mod fn_with_type_in_body {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn boop() -> f32 {
        Point::origin().total()
    }
}

/// A fn X that calls a fn Y, where Y uses the changed type in its
/// body. In this case, the effects of the change should be contained
/// to Y; X should not have to be rebuilt, nor should it need to be
/// typechecked again.
mod call_fn_with_type_in_body {
    use fn_with_type_in_body;

    #[rustc_clean(label="TypeckItemBody", cfg="rpass2")]
    pub fn bip() -> f32 {
        fn_with_type_in_body::boop()
    }
}

/// A fn item that makes an instance of `Point` but does not invoke methods
mod fn_make_struct {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn make_origin(p: Point) -> Point {
        Point { ..p }
    }
}

/// A fn item that reads fields from `Point` but does not invoke methods
mod fn_read_field {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn get_x(p: Point) -> f32 {
        p.x
    }
}

/// A fn item that writes to a field of `Point` but does not invoke methods
mod fn_write_field {
    use point::Point;

    #[rustc_dirty(label="TypeckItemBody", cfg="rpass2")]
    pub fn inc_x(p: &mut Point) {
        p.x += 1.0;
    }
}

fn main() {
}
