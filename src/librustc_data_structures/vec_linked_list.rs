use crate::indexed_vec::{Idx, IndexVec};

pub fn iter<Ls>(
    first: Option<Ls::LinkIndex>,
    links: &'a Ls,
) -> impl Iterator<Item = Ls::LinkIndex> + 'a
where
    Ls: Links,
{
    VecLinkedListIterator {
        links,
        current: first,
    }
}

pub struct VecLinkedListIterator<Ls>
where
    Ls: Links,
{
    links: Ls,
    current: Option<Ls::LinkIndex>,
}

impl<Ls> Iterator for VecLinkedListIterator<Ls>
where
    Ls: Links,
{
    type Item = Ls::LinkIndex;

    fn next(&mut self) -> Option<Ls::LinkIndex> {
        if let Some(c) = self.current {
            self.current = <Ls as Links>::next(&self.links, c);
            Some(c)
        } else {
            None
        }
    }
}

pub trait Links {
    type LinkIndex: Copy;

    fn next(links: &Self, index: Self::LinkIndex) -> Option<Self::LinkIndex>;
}

impl<Ls> Links for &Ls
where
    Ls: Links,
{
    type LinkIndex = Ls::LinkIndex;

    fn next(links: &Self, index: Ls::LinkIndex) -> Option<Ls::LinkIndex> {
        <Ls as Links>::next(links, index)
    }
}

pub trait LinkElem {
    type LinkIndex: Copy;

    fn next(elem: &Self) -> Option<Self::LinkIndex>;
}

impl<L, E> Links for IndexVec<L, E>
where
    E: LinkElem<LinkIndex = L>,
    L: Idx,
{
    type LinkIndex = L;

    fn next(links: &Self, index: L) -> Option<L> {
        <E as LinkElem>::next(&links[index])
    }
}
