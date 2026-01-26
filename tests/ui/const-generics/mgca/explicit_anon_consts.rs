#![feature(generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]
// library crates exercise weirder code paths around
// DefIds which were created for const args.
#![crate_type = "lib"]

struct Foo<const N: usize>;

type Adt1<const N: usize> = Foo<N>;
type Adt2<const N: usize> = Foo<{ N }>;
type Adt3<const N: usize> = Foo<const { N }>;
//~^ ERROR: generic parameters may not be used in const operations
type Adt4<const N: usize> = Foo<{ 1 + 1 }>;
//~^ ERROR: complex const arguments must be placed inside of a `const` block
type Adt5<const N: usize> = Foo<const { 1 + 1 }>;

type Arr<const N: usize> = [(); N];
type Arr2<const N: usize> = [(); { N }];
type Arr3<const N: usize> = [(); const { N }];
//~^ ERROR: generic parameters may not be used in const operations
type Arr4<const N: usize> = [(); 1 + 1];
//~^ ERROR: complex const arguments must be placed inside of a `const` block
type Arr5<const N: usize> = [(); const { 1 + 1 }];

fn repeats<const N: usize>() -> [(); N] {
    let _1 = [(); N];
    let _2 = [(); { N }];
    let _3 = [(); const { N }];
    //~^ ERROR: generic parameters may not be used in const operations
    let _4 = [(); 1 + 1];
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    let _5 = [(); const { 1 + 1 }];
    let _6: [(); const { N }] = todo!();
    //~^ ERROR: generic parameters may not be used in const operations
}

#[type_const]
const ITEM1<const N: usize>: usize = N;
#[type_const]
const ITEM2<const N: usize>: usize = { N };
#[type_const]
const ITEM3<const N: usize>: usize = const { N };
//~^ ERROR: generic parameters may not be used in const operations
#[type_const]
const ITEM4<const N: usize>: usize = { 1 + 1 };
//~^ ERROR: complex const arguments must be placed inside of a `const` block
#[type_const]
const ITEM5<const N: usize>: usize = const { 1 + 1};

trait Trait {
    #[type_const]
    const ASSOC: usize;
}

fn ace_bounds<
    const N: usize,
    // We skip the T1 case because it doesn't resolve
    // T1: Trait<ASSOC = N>,
    T2: Trait<ASSOC = { N }>,
    T3: Trait<ASSOC = const { N }>,
    //~^ ERROR: generic parameters may not be used in const operations
    T4: Trait<ASSOC = { 1 + 1 }>,
    //~^ ERROR: complex const arguments must be placed inside of a `const` block
    T5: Trait<ASSOC = const { 1 + 1 }>,
>() {}

struct Default1<const N: usize, const M: usize = N>;
struct Default2<const N: usize, const M: usize = { N }>;
struct Default3<const N: usize, const M: usize = const { N }>;
//~^ ERROR: generic parameters may not be used in const operations
struct Default4<const N: usize, const M: usize = { 1 + 1 }>;
//~^ ERROR: complex const arguments must be placed inside of a `const` block
struct Default5<const N: usize, const M: usize = const { 1 + 1}>;
