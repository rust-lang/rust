#![feature(const_generics_defaults)]
struct Foo<const N: usize = 1, const N2: usize = 2>;
struct Bar<const N: usize, const N2: usize = { N + 1 }>;
struct Lots<
    const N1BlahFooUwU: usize = { 10 + 28 + 1872 / 10 * 3 },
    const N2SecondParamOhmyyy: usize = { N1BlahFooUwU / 2 + 10 * 2 },
>;
struct NamesRHard<const N: usize = { 1 + 1 + 1 + 1 + 1 + 1 }>;
struct FooBar<
    const LessThan100ButClose: usize = {
        1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1
    },
>;
struct FooBarrrrrrrr<
    const N: usize = {
        13478234326456456444323871
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
            + 1
    },
>;
