pub mod author;
pub mod dump_hir;
pub mod format_args_collector;
#[cfg(feature = "internal")]
pub mod internal_lints;

// ==================================================================
// Configuration
// ==================================================================

// Shamelessly stolen from find_all (https://github.com/nectariner/find_all)
pub trait FindAll: Iterator + Sized {
    fn find_all<P>(&mut self, predicate: P) -> Option<Vec<usize>>
    where
        P: FnMut(&Self::Item) -> bool;
}

impl<I> FindAll for I
where
    I: Iterator,
{
    fn find_all<P>(&mut self, mut predicate: P) -> Option<Vec<usize>>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        let mut occurences = Vec::<usize>::default();
        for (index, element) in self.enumerate() {
            if predicate(&element) {
                occurences.push(index);
            }
        }

        match occurences.len() {
            0 => None,
            _ => Some(occurences),
        }
    }
}
