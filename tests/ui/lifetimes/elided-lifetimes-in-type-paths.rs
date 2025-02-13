#![deny(elided_lifetimes_in_paths)]

// Most of the time, we focus on elided lifetimes in function
// signatures, but they can also appear in other places! The original
// version of this lint handled all these cases in one location, but
// it's desired that the one lint actually be multiple.

struct ContainsLifetime<'a>(&'a u8);

impl<'a> ContainsLifetime<'a> {
    fn foo() {}
}

fn use_via_turbofish<T>() {}

trait UseViaTrait {
    fn use_it(&self) {}
}

impl UseViaTrait for ContainsLifetime<'_> {}

// ==========

static USE_VIA_STATIC: ContainsLifetime = ContainsLifetime(&42);
//~^ ERROR hidden lifetime parameters

fn main() {
    use_via_turbofish::<ContainsLifetime>();
    //~^ ERROR hidden lifetime parameters

    let _use_via_binding: ContainsLifetime;
    //~^ ERROR hidden lifetime parameters

    _ = <ContainsLifetime as UseViaTrait>::use_it;
    //~^ ERROR hidden lifetime parameters

    ContainsLifetime::foo();
}
