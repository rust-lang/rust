// Checks that applied type arguments of enums, and aliases to them, are respected.
// For example, `Self` is never a type constructor. Therefore, no types can be applied to it.
//
// We also check that the variant to an type-aliased enum cannot be type applied whether
// that alias is generic or monomorphic.

enum Enum<T> { TSVariant(T), SVariant { v: T }, UVariant }
type Alias<T> = Enum<T>;
type AliasFixed = Enum<()>;

impl<T> Enum<T> {
    fn ts_variant() {
        Self::TSVariant(());
        //~^ ERROR mismatched types [E0308]
        Self::TSVariant::<()>(());
        //~^ ERROR type arguments are not allowed for this type [E0109]
        Self::<()>::TSVariant(());
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR mismatched types [E0308]
        Self::<()>::TSVariant::<()>(());
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR type arguments are not allowed for this type [E0109]
    }

    fn s_variant() {
        Self::SVariant { v: () };
        //~^ ERROR mismatched types [E0308]
        Self::SVariant::<()> { v: () };
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR mismatched types [E0308]
        Self::<()>::SVariant { v: () };
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR mismatched types [E0308]
        Self::<()>::SVariant::<()> { v: () };
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR mismatched types [E0308]
    }

    fn u_variant() {
        Self::UVariant::<()>;
        //~^ ERROR type arguments are not allowed for this type [E0109]
        Self::<()>::UVariant;
        //~^ ERROR type arguments are not allowed for this type [E0109]
        Self::<()>::UVariant::<()>;
        //~^ ERROR type arguments are not allowed for this type [E0109]
        //~| ERROR type arguments are not allowed for this type [E0109]
    }
}

fn main() {
    // Tuple struct variant

    Enum::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed for this type [E0109]

    Alias::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed for this type [E0109]
    Alias::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed for this type [E0109]

    AliasFixed::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed for this type [E0109]
    AliasFixed::<()>::TSVariant(());
    //~^ ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]
    AliasFixed::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed for this type [E0109]
    //~| ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]

    // Struct variant

    Enum::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed for this type [E0109]

    Alias::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed for this type [E0109]
    Alias::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed for this type [E0109]

    AliasFixed::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed for this type [E0109]
    AliasFixed::<()>::SVariant { v: () };
    //~^ ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]
    AliasFixed::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed for this type [E0109]
    //~| ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]

    // Unit variant

    Enum::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed for this type [E0109]

    Alias::UVariant::<()>;
    //~^ ERROR type arguments are not allowed for this type [E0109]
    Alias::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed for this type [E0109]

    AliasFixed::UVariant::<()>;
    //~^ ERROR type arguments are not allowed for this type [E0109]
    AliasFixed::<()>::UVariant;
    //~^ ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]
    AliasFixed::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed for this type [E0109]
    //~| ERROR this type alias takes 0 type arguments but 1 type argument was supplied [E0107]
}
