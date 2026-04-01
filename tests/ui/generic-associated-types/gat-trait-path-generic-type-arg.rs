trait Foo {
    type F<'a>;

    fn identity<'a>(t: &'a Self::F<'a>) -> &'a Self::F<'a> { t }
}

impl <T, T1> Foo for T {
    //~^ ERROR: the type parameter `T1` is not constrained
    type F<T1> = &[u8];
      //~^ ERROR: the name `T1` is already used for
      //~| ERROR: `&` without an explicit lifetime name cannot be used here
      //~| ERROR: has 1 type parameter but its trait declaration has 0 type parameters
}

fn main() {}
