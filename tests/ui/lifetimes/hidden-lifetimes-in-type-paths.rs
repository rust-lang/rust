#![deny(hidden_lifetimes_in_type_paths)]

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
