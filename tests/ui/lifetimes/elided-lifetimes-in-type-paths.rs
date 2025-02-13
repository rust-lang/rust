#![deny(elided_lifetimes_in_paths)]

// Most of the time, we focus on elided lifetimes in function
// signatures, but they can also appear in other places! The original
// version of this lint handled all these cases in one location, but
// it's desired that the one lint actually be multiple.

struct ContainsLifetime<'a>(&'a u8);

impl<'a> ContainsLifetime<'a> {
    fn use_it() {}
}

struct ContainsLifetimeAndType<'a, T>(&'a T);

impl<'a, T> ContainsLifetimeAndType<'a, T> {
    fn use_it() {}
}

fn use_via_turbofish<T>() {}

trait UseViaTrait {
    fn use_it() {}
}

impl UseViaTrait for ContainsLifetime<'_> {}

trait TraitWithLifetime<'a> {
    fn use_it() {}
}

impl<'a> TraitWithLifetime<'a> for u8 {}

enum EnumWithType<T> {
    VariantStructLike { v: T },
    VariantTupleLike(T),
    VariantUnit,
}

type TypeAliasWithLifetime<'a> = EnumWithType<&'a u8>;

// ==========

static USE_VIA_STATIC: ContainsLifetime = ContainsLifetime(&42);
//~^ ERROR hidden lifetime parameters

fn main() {
    use_via_turbofish::<ContainsLifetime>();
    //~^ ERROR hidden lifetime parameters

    let _use_via_binding: ContainsLifetime;
    //~^ ERROR hidden lifetime parameters

    <ContainsLifetime as UseViaTrait>::use_it();
    //~^ ERROR hidden lifetime parameters

    <ContainsLifetime>::use_it();
    //~^ ERROR hidden lifetime parameters

    ContainsLifetime::use_it();

    ContainsLifetimeAndType::<u8>::use_it();

    <u8 as TraitWithLifetime>::use_it();
}

impl TypeAliasWithLifetime<'_> {
    fn use_via_match(self) {
        match self {
            TypeAliasWithLifetime::VariantStructLike { .. } => (),
            TypeAliasWithLifetime::VariantTupleLike(_) => (),
            TypeAliasWithLifetime::VariantUnit => (),
        }
    }

    fn use_via_create(v: u8) -> Self {
        match v {
            0 => TypeAliasWithLifetime::VariantStructLike { v: &42 },
            1 => TypeAliasWithLifetime::VariantTupleLike(&42),
            _ => TypeAliasWithLifetime::VariantUnit,
        }
    }
}
