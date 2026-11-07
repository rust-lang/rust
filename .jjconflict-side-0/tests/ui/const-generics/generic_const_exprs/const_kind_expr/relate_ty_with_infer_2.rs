//@ check-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
use std::marker::PhantomData;

pub struct Equal<const T: usize, const R: usize>();
pub trait True {}
impl<const T: usize> True for Equal<T, T> {}

// replacement for generativity
pub struct Id<'id>(PhantomData<fn(&'id ()) -> &'id ()>);
pub struct Guard<'id>(Id<'id>);
fn make_guard<'id>(i: &'id Id<'id>) -> Guard<'id> {
    Guard(Id(PhantomData))
}

impl<'id> Into<Id<'id>> for Guard<'id> {
    fn into(self) -> Id<'id> {
        self.0
    }
}

pub struct Arena<'life> {
    bytes: *mut [u8],
    //bitmap: RefCell<RoaringBitmap>,
    _token: PhantomData<Id<'life>>,
}

#[repr(transparent)]
pub struct Item<'life, T> {
    data: T,
    _phantom: PhantomData<Id<'life>>,
}

#[repr(transparent)]
pub struct Token<'life, 'borrow, 'compact, 'reborrow, T>
where
    'life: 'reborrow,
    T: Tokenize<'life, 'borrow, 'compact, 'reborrow>,
{
    //ptr: *mut <T as Tokenize>::Tokenized,
    ptr: core::ptr::NonNull<T::Tokenized>,
    _phantom: PhantomData<Id<'life>>,
    _compact: PhantomData<&'borrow Guard<'compact>>,
    _result: PhantomData<&'reborrow T::Untokenized>,
}

impl<'life> Arena<'life> {
    pub fn tokenize<'before, 'compact, 'borrow, 'reborrow, T, U>(
        &self,
        guard: &'borrow Guard<'compact>,
        item: Item<'life, &'before mut T>,
    ) -> Token<'life, 'borrow, 'compact, 'reborrow, U>
    where
        T: Tokenize<'life, 'borrow, 'compact, 'reborrow, Untokenized = U>,
        T::Untokenized: Tokenize<'life, 'borrow, 'compact, 'reborrow>,
        Equal<{ core::mem::size_of::<T>() }, { core::mem::size_of::<U>() }>: True,
        'compact: 'borrow,
        'life: 'reborrow,
        'life: 'compact,
        'life: 'borrow,
        // 'borrow: 'before ??
    {
        let dst = item.data as *mut T as *mut T::Tokenized;
        Token {
            ptr: core::ptr::NonNull::new(dst as *mut _).unwrap(),
            _phantom: PhantomData,
            _compact: PhantomData,
            _result: PhantomData,
        }
    }
}

pub trait Tokenize<'life, 'borrow, 'compact, 'reborrow>
where
    'compact: 'borrow,
    'life: 'reborrow,
    'life: 'borrow,
    'life: 'compact,
{
    type Tokenized;
    type Untokenized;
    const TO: fn(&Arena<'life>, &'borrow Guard<'compact>, Self) -> Self::Tokenized;
    const FROM: fn(&'reborrow Arena<'life>, Self::Tokenized) -> Self::Untokenized;
}

macro_rules! tokenize {
    ($to:expr, $from:expr) => {
        const TO: fn(&Arena<'life>, &'borrow Guard<'compact>, Self) -> Self::Tokenized = $to;
        const FROM: fn(&'reborrow Arena<'life>, Self::Tokenized) -> Self::Untokenized = $from;
    };
}

struct Foo<'life, 'borrow>(Option<Item<'life, &'borrow mut Bar>>);
struct TokenFoo<'life, 'borrow, 'compact, 'reborrow>(
    Option<Token<'life, 'borrow, 'compact, 'reborrow, Bar>>,
);
struct Bar(u8);

impl<'life, 'before, 'borrow, 'compact, 'reborrow> Tokenize<'life, 'borrow, 'compact, 'reborrow>
    for Foo<'life, 'before>
where
    'compact: 'borrow,
    'life: 'reborrow,
    'life: 'borrow,
    'life: 'compact,
{
    type Tokenized = TokenFoo<'life, 'borrow, 'compact, 'reborrow>;
    type Untokenized = Foo<'life, 'reborrow>;
    tokenize!(foo_to, foo_from);
}

impl<'life, 'borrow, 'compact, 'reborrow> Tokenize<'life, 'borrow, 'compact, 'reborrow> for Bar
where
    'compact: 'borrow,
    'life: 'reborrow,
    'life: 'borrow,
    'life: 'compact,
{
    type Tokenized = Bar;
    type Untokenized = Bar;
    tokenize!(bar_to, bar_from);
}

fn bar_to<'life, 'borrow, 'compact>(
    arena: &Arena<'life>,
    guard: &'borrow Guard<'compact>,
    s: Bar,
) -> Bar {
    s
}
fn bar_from<'life, 'reborrow>(arena: &'reborrow Arena<'life>, s: Bar) -> Bar {
    s
}

fn foo_to<'life, 'borrow, 'compact, 'reborrow, 'before>(
    arena: &'before Arena<'life>,
    guard: &'borrow Guard<'compact>,
    s: Foo<'life, 'before>,
) -> TokenFoo<'life, 'borrow, 'compact, 'reborrow> {
    let Foo(bar) = s;
    TokenFoo(bar.map(|bar| arena.tokenize(guard, bar)))
}
fn foo_from<'life, 'borrow, 'compact, 'reborrow>(
    arena: &'reborrow Arena<'life>,
    s: TokenFoo<'life, 'borrow, 'compact, 'reborrow>,
) -> Foo<'life, 'reborrow> {
    Foo(s.0.map(|bar| panic!()))
}

fn main() {}
