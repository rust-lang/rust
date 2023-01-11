// A first "spike" for incremental compilation: here, we change the
// content of the `make` function, and we find that we can reuse the
// `y` module entirely (but not the `x` module).

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="spike", cfg="rpass2")]
#![rustc_partition_codegened(module="spike-x", cfg="rpass2")]
#![rustc_partition_reused(module="spike-y", cfg="rpass2")]

mod x {
    pub struct X {
        x: u32, y: u32,
    }

    #[cfg(rpass1)]
    fn make() -> X {
        X { x: 22, y: 0 }
    }

    #[cfg(rpass2)]
    fn make() -> X {
        X { x: 11, y: 11 }
    }

    pub fn new() -> X {
        make()
    }

    pub fn sum(x: &X) -> u32 {
        x.x + x.y
    }
}

mod y {
    use x;

    pub fn assert_sum() -> bool {
        let x = x::new();
        x::sum(&x) == 22
    }
}

pub fn main() {
    y::assert_sum();
}
