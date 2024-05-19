//! Strategies for `u128` and `i128`, since proptest doesn't provide them for the wasm target.

macro_rules! impl_num {
    { $name:ident } => {
        pub(crate) mod $name {
            type InnerStrategy = crate::array::UniformArrayStrategy<proptest::num::u64::Any, [u64; 2]>;
            use proptest::strategy::{Strategy, ValueTree, NewTree};


            #[must_use = "strategies do nothing unless used"]
            #[derive(Clone, Copy, Debug)]
            pub struct Any {
                strategy: InnerStrategy,
            }

            pub struct BinarySearch {
                inner: <InnerStrategy as Strategy>::Tree,
            }

            impl ValueTree for BinarySearch {
                type Value = $name;

                fn current(&self) -> $name {
                    unsafe { core::mem::transmute(self.inner.current()) }
                }

                fn simplify(&mut self) -> bool {
                    self.inner.simplify()
                }

                fn complicate(&mut self) -> bool {
                    self.inner.complicate()
                }
            }

            impl Strategy for Any {
                type Tree = BinarySearch;
                type Value = $name;

                fn new_tree(&self, runner: &mut proptest::test_runner::TestRunner) -> NewTree<Self> {
                    Ok(BinarySearch { inner: self.strategy.new_tree(runner)? })
                }
            }

            pub const ANY: Any = Any { strategy: InnerStrategy::new(proptest::num::u64::ANY) };
        }
    }
}

impl_num! { u128 }
impl_num! { i128 }
