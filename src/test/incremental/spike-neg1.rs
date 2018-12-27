// A variant of the first "spike" test that serves to test the
// `rustc_partition_reused` and `rustc_partition_codegened` tests.
// Here we change and say that the `x` module will be reused (when in
// fact it will not), and then indicate that the test itself
// should-fail (because an error will be reported, and hence the
// revision rpass2 will not compile, despite being named rpass).

// revisions:rpass1 rpass2
// should-fail

#![feature(rustc_attrs)]

#![rustc_partition_reused(module="spike_neg1", cfg="rpass2")]
#![rustc_partition_reused(module="spike_neg1-x", cfg="rpass2")] // this is wrong!
#![rustc_partition_reused(module="spike_neg1-y", cfg="rpass2")]

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
