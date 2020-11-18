// revisions: full min
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(min, feature(min_const_generics))]

mod m {
    pub const P: usize = 0;
}

const Q: usize = 0;

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
    let _: foo!(Q);
    let _: foo!(m::P);
    let _: bar!(N);
    let _: bar!({ N }); //[min]~ ERROR generic parameters may not
    let _: bar!(Q);
    let _: bar!(m::P);
    let _: baz!(N);
    let _: baz!({ N });
    let _: baz!({{ N }}); //[min]~ ERROR generic parameters may not
    let _: baz!(Q);
    let _: baz!({ m::P });
    let _: baz!(m::P); //~ ERROR expressions must be enclosed in braces
    let _: biz!(N);
    let _: biz!({ N }); //[min]~ ERROR generic parameters may not
    let _: biz!(Q);
    let _: biz!(m::P);
    let _: foo!(3);
    let _: foo!({ 3 });
    let _: foo!({{ 3 }});
    let _: bar!(3);
    let _: bar!({ 3 });
    let _: baz!(3);
    let _: baz!({ 3 });
    let _: baz!({{ 3 }});
    let _: biz!(3);
    let _: biz!({ 3 });
    let _: foo!(10 + 7);
    let _: foo!({ 10 + 7 });
    let _: foo!({{ 10 + 7 }});
    let _: bar!(10 + 7);
    let _: bar!({ 10 + 7 });
    let _: baz!(10 + 7); //~ ERROR expressions must be enclosed in braces
    let _: baz!({ 10 + 7 });
    let _: baz!({{ 10 + 7 }});
    let _: biz!(10 + 7);
    let _: biz!({ 10 + 7 });
}

fn main() {
    test::<3>();
}
