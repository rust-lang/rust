#![crate_type="lib"]

struct Nested<K>(K);

fn should_error<T>() where T : Into<&u32> {}
//~^ ERROR `&` without an explicit lifetime name cannot be used here [E0637]

trait X<'a, K: 'a> {
    fn foo<'b, L: X<&'b Nested<K>>>();
    //~^ ERROR missing lifetime specifier [E0106]
}

fn bar<'b, L: X<&'b Nested<i32>>>(){}
//~^ ERROR missing lifetime specifier [E0106]
