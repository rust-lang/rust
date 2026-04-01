type Plain = Foo<Item, Item::Item, Item: Bound, Item = Item>;
type GenericArgs = Foo<Item<T>, Item::<T>, Item<T>: Bound, Item::<T>: Bound, Item<T> = Item, Item::<T> = Item>;
type ParenthesizedArgs = Foo<Item(T), Item::(T), Item(T): Bound, Item::(T): Bound, Item(T) = Item, Item::(T) = Item>;
type RTN = Foo<Item(..), Item(..), Item(..): Bound, Item(..): Bound, Item(..) = Item, Item(..) = Item>;
