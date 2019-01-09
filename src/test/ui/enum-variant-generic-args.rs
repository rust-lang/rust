#![feature(type_alias_enum_variants)]

enum Enum<T> { TSVariant(T), SVariant { v: T } }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

impl<T> Enum<T> {
    fn ts_variant() {
        Self::TSVariant(());
        //~^ ERROR mismatched types [E0308]
        Self::TSVariant::<()>(());
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        Self::<()>::TSVariant(());
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^ ERROR mismatched types [E0308]
        Self::<()>::TSVariant::<()>(());
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^ ERROR type arguments are not allowed on this entity [E0109]
    }

    fn s_variant() {
        Self::SVariant { v: () };
        //~^ ERROR mismatched types [E0308]
        Self::SVariant::<()> { v: () };
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^ ERROR mismatched types [E0308]
        Self::<()>::SVariant { v: () };
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^ ERROR mismatched types [E0308]
        Self::<()>::SVariant::<()> { v: () };
        //~^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^ ERROR type arguments are not allowed on this entity [E0109]
        //~^^^ ERROR mismatched types [E0308]
    }
}

fn main() {
    // Tuple struct variant

    Enum::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this entity [E0109]

    Alias::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    Alias::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this entity [E0109]

    AliasFixed::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    AliasFixed::<()>::TSVariant(());
    //~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
    AliasFixed::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    //~^^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]

    // Struct variant

    Enum::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this entity [E0109]

    Alias::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    Alias::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this entity [E0109]

    AliasFixed::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    AliasFixed::<()>::SVariant { v: () };
    //~^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
    AliasFixed::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this entity [E0109]
    //~^^ ERROR wrong number of type arguments: expected 0, found 1 [E0107]
}
