pub use self::imp::*;

#[cfg(feature = "perf-literal")]
mod imp;

#[allow(missing_docs)]
#[cfg(not(feature = "perf-literal"))]
mod imp {
    use regex_syntax::hir::literal::Seq;

    #[derive(Clone, Debug)]
    pub struct LiteralSearcher(());

    impl LiteralSearcher {
        pub fn empty() -> Self {
            LiteralSearcher(())
        }

        pub fn prefixes(_: Seq) -> Self {
            LiteralSearcher(())
        }

        pub fn suffixes(_: Seq) -> Self {
            LiteralSearcher(())
        }

        pub fn complete(&self) -> bool {
            false
        }

        pub fn find(&self, _: &[u8]) -> Option<(usize, usize)> {
            unreachable!()
        }

        pub fn find_start(&self, _: &[u8]) -> Option<(usize, usize)> {
            unreachable!()
        }

        pub fn find_end(&self, _: &[u8]) -> Option<(usize, usize)> {
            unreachable!()
        }

        pub fn is_empty(&self) -> bool {
            true
        }

        pub fn len(&self) -> usize {
            0
        }

        pub fn approximate_size(&self) -> usize {
            0
        }
    }
}
