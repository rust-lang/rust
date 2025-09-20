// Regression test for #146706
// Check that number of path segments of enum type and alias does not break
// suggestions for the error that variant of type-aliased enum cannot have
// generic type argument.

mod foo {
    pub enum Enum<T> { TSVariant(T), SVariant { v: T }, UVariant }
    pub type Alias<T> = Enum<T>;
    pub type AliasFixed = Enum<()>;
}

fn main() {
    // Tuple struct variant

    foo::Enum::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on tuple variant `TSVariant` [E0109]

    foo::Alias::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::Alias::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this type [E0109]

    foo::AliasFixed::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::AliasFixed::<()>::TSVariant(());
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    foo::AliasFixed::<()>::TSVariant::<()>(());
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]

    // Struct variant

    foo::Enum::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on variant `SVariant` [E0109]

    foo::Alias::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::Alias::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this type [E0109]

    foo::AliasFixed::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::AliasFixed::<()>::SVariant { v: () };
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    foo::AliasFixed::<()>::SVariant::<()> { v: () };
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]

    // Unit variant

    foo::Enum::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on unit variant `UVariant` [E0109]

    foo::Alias::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::Alias::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]

    foo::AliasFixed::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    foo::AliasFixed::<()>::UVariant;
    //~^ ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
    foo::AliasFixed::<()>::UVariant::<()>;
    //~^ ERROR type arguments are not allowed on this type [E0109]
    //~| ERROR type alias takes 0 generic arguments but 1 generic argument was supplied [E0107]
}
