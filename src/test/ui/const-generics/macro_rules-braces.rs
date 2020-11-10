// revisions: full min
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

fn test<const N: usize>() {
    struct Foo<const M: usize>;
    macro_rules! foo {
        ($x:expr) => {
            [u8; $x] //[full]~ ERROR constant expression depends
        }
    }
    macro_rules! bar {
        ($x:expr) => {
            [u8; { $x }] //[full]~ ERROR constant expression depends
        }
    }
    macro_rules! baz {
        ( $x:expr) => {
            Foo<$x> //[full]~ ERROR constant expression depends
        }
    }
    macro_rules! biz {
        ($x:expr) => {
            Foo<{ $x }> //[full]~ ERROR constant expression depends
        };
    }

    let _: foo!(N);
    let _: foo!({ N });
    let _: foo!({{ N }}); //[min]~ ERROR generic parameters may not
    let _: bar!(N);
    let _: bar!({ N }); //[min]~ ERROR generic parameters may not
    let _: baz!(N); //~ ERROR expressions must be enclosed in braces
    let _: baz!({ N });
    let _: baz!({{ N }}); //[min]~ ERROR generic parameters may not
    let _: biz!(N);
    let _: biz!({ N }); //[min]~ ERROR generic parameters may not
}

fn main() {
    test::<3>();
}
