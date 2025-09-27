// Regression test for #146706
// Check that elided lifetime argument does not break suggestions for the error
// that variant of type-aliased enum cannot have generic type argument.

use std::marker::PhantomData;

enum Enum<'a, T> {
    TSVariant(PhantomData<&'a T>),
    SVariant { phantom: PhantomData<&'a T> },
    UVariant,
}
type Alias<'a, T> = Enum<'a, T>;
type AliasFixed<'a> = Enum<'a, ()>;

impl<'a, T> Enum<'a, T> {
    fn ts_variant() {
        Self::TSVariant(PhantomData);
        Self::TSVariant::<()>(PhantomData);
        //~^ ERROR type arguments are not allowed on this type [E0109]
        Self::<()>::TSVariant(PhantomData);
        //~^ ERROR type arguments are not allowed on self type [E0109]
        Self::<()>::TSVariant::<()>(PhantomData);
        //~^ ERROR type arguments are not allowed on self type [E0109]
        //~| ERROR type arguments are not allowed on this type [E0109]
    }

    fn s_variant() {
        Self::SVariant { phantom: PhantomData };
        Self::SVariant::<()> { phantom: PhantomData };
        //~^ ERROR type arguments are not allowed on this type [E0109]
        Self::<()>::SVariant { phantom: PhantomData };
        //~^ ERROR type arguments are not allowed on self type [E0109]
        Self::<()>::SVariant::<()> { phantom: PhantomData };
        //~^ ERROR type arguments are not allowed on self type [E0109]
        //~| ERROR type arguments are not allowed on this type [E0109]
    }

    fn u_variant() {
        Self::UVariant::<()>;
        //~^ ERROR type arguments are not allowed on this type [E0109]
        Self::<()>::UVariant;
        //~^ ERROR type arguments are not allowed on self type [E0109]
        Self::<()>::UVariant::<()>;
        //~^ ERROR type arguments are not allowed on self type [E0109]
        //~| ERROR type arguments are not allowed on this type [E0109]
    }
}

fn main() {
    // Tuple struct variant

    Enum::<()>::TSVariant::<()>(PhantomData);
    //~^ ERROR type arguments are not allowed on tuple variant `TSVariant` [E0109]

    Alias::TSVariant::<()>(PhantomData);
    //~^ ERROR type arguments are not allowed on this type [E0109]
    Alias::<()>::TSVariant::<()>(PhantomData);
    //~^ ERROR type arguments are not allowed on this type [E0109]

    AliasFixed::TSVariant::<()>(PhantomData);
    //~^ ERROR type arguments are not allowed on this type [E0109]
    AliasFixed::<()>::TSVariant(PhantomData);
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    AliasFixed::<()>::TSVariant::<()>(PhantomData);
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]

    // Struct variant

    Enum::<()>::SVariant::<()> { phantom: PhantomData };
    //~^ ERROR type arguments are not allowed on variant `SVariant` [E0109]
    //~| ERROR enum takes 1 lifetime argument but 0 lifetime arguments were supplied [E0107]

    Alias::SVariant::<()> { phantom: PhantomData };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    Alias::<()>::SVariant::<()> { phantom: PhantomData };
    //~^ ERROR type arguments are not allowed on this type [E0109]

    AliasFixed::SVariant::<()> { phantom: PhantomData };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    AliasFixed::<()>::SVariant { phantom: PhantomData };
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    AliasFixed::<()>::SVariant::<()> { phantom: PhantomData };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]

    // Unit variant

    Enum::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on unit variant `UVariant` [E0109]

    Alias::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    Alias::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]

    AliasFixed::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    AliasFixed::<()>::UVariant;
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    AliasFixed::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
}
