//@ known-bug: #123917
//@ compile-flags: -Zmir-opt-level=5 -Zpolymorphize=on

use std::marker::PhantomData;

pub struct Id<'id>();

pub struct Item<'life, T> {
    data: T,
}

pub struct Token<'life, 'borrow, 'compact, 'reborrow, T>
where
    'life: 'reborrow,
    T: Tokenize,
{
    ptr: *mut <T as Tokenize>::Tokenized,
    ptr: core::ptr::NonNull<T::Tokenized>,
    _phantom: PhantomData<Id<'life>>,
}

impl<'life> Arena<'life> {
    pub fn tokenize<'before, 'compact, 'borrow, 'reborrow, T, U>(
        item: Item<'life, &'before mut T>,
    ) -> Token<'life, 'borrow, 'compact, 'reborrow, U>
    where
        T: Tokenize<'life, 'borrow, 'compact, 'reborrow, Untokenized = U>,
        T::Untokenized: Tokenize<'life, 'borrow, 'compact, 'reborrow>,
    {
        let dst = item.data as *mut T as *mut T::Tokenized;
        Token {
            ptr: core::ptr::NonNull::new(dst as *mut _).unwrap(),
            _phantom: PhantomData,
        }
    }
}

pub trait Tokenize {
    type Tokenized;
    type Untokenized;
}
