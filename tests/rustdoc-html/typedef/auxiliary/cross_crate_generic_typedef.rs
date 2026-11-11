pub struct InlineOne<A> {
   pub inline: A,
   #[doc(hidden)]
   pub hidden: A,
}

pub type InlineU64 = InlineOne<u64>;

pub enum GenericEnum<T> {
   Variant(T),
   Variant2(T, T),
   #[doc(hidden)]
   Hidden(T),
}
