struct Struct<const N: usize = { Self; 10 }>;
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

enum Enum<const N: usize = { Self; 10 }> { }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

union Union<const N: usize = { Self; 10 }> { not_empty: () }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

fn main() {
    let _: Struct;
    let _: Enum;
    let _: Union;
}
