#![feature(type_alias_enum_variants)]

enum Enum<T> { Variant(T) }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

impl<T> Enum<T> {
    fn foo() {
        Self::Variant::<()>(());
        //~^ ERROR type parameters are not allowed on this type [E0109]
        Self::<()>::Variant(());
        //~^ ERROR type parameters are not allowed on this type [E0109]
        Self::<()>::Variant::<()>(());
        //~^ ERROR type parameters are not allowed on this type [E0109]
        //~^^ ERROR type parameters are not allowed on this type [E0109]
    }
}

fn main() {
    Enum::<()>::Variant::<()>(());
    //~^ ERROR type parameters are not allowed on this type [E0109]

    Alias::Variant::<()>(());
    //~^ ERROR type parameters are not allowed on this type [E0109]
    Alias::<()>::Variant::<()>(());
    //~^ ERROR type parameters are not allowed on this type [E0109]

    AliasFixed::Variant::<()>(());
    //~^ ERROR type parameters are not allowed on this type [E0109]
    AliasFixed::<()>::Variant(());
    //~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
    AliasFixed::<()>::Variant::<()>(());
    //~^ ERROR type parameters are not allowed on this type [E0109]
    //~^^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
}
