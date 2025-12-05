pub struct InlineOne<A> {
   pub inline: A
}

pub type InlineU64 = InlineOne<u64>;

pub enum GenericEnum<T> {
   Variant(T),
   Variant2(T, T),
}
