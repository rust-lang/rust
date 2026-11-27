//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Generic<const K: u64>;

struct ConstU64<const K: u64>;

impl<const K: u64> Generic<K>
where
    ConstU64<{ K - 1 }>: ,
{
    fn foo(self) -> u64 {
        K
    }
}

impl<const K: u64> Generic<K>
where
    ConstU64<{ K - 1 }>: ,
    ConstU64<{ K + 1 }>: ,
    ConstU64<{ K + 1 - 1 }>: ,
{
    fn bar(self) -> u64 {
        let x: Generic<{ K + 1 }> = Generic;
        x.foo()
    }
}

fn main() {
    assert_eq!((Generic::<10>).bar(), 11);
}

// Test that the ``ConstU64<{ K + 1 - 1}>`` bound on ``bar``'s impl block satisfies the
// ``ConstU64<{K - 1}>`` bound on ``foo``'s impl block
