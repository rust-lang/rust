// run-pass
macro_rules! fooN {
    ($cur:ident $prev:ty) => {
        #[allow(dead_code)]
        enum $cur {
            Empty,
            First($prev),
            Second($prev),
            Third($prev),
            Fourth($prev),
        }
    }
}

fooN!(Foo0 ());
fooN!(Foo1 Foo0);
fooN!(Foo2 Foo1);
fooN!(Foo3 Foo2);
fooN!(Foo4 Foo3);
fooN!(Foo5 Foo4);
fooN!(Foo6 Foo5);
fooN!(Foo7 Foo6);
fooN!(Foo8 Foo7);
fooN!(Foo9 Foo8);
fooN!(Foo10 Foo9);
fooN!(Foo11 Foo10);
fooN!(Foo12 Foo11);
fooN!(Foo13 Foo12);
fooN!(Foo14 Foo13);
fooN!(Foo15 Foo14);
fooN!(Foo16 Foo15);
fooN!(Foo17 Foo16);
fooN!(Foo18 Foo17);
fooN!(Foo19 Foo18);
fooN!(Foo20 Foo19);
fooN!(Foo21 Foo20);
fooN!(Foo22 Foo21);
fooN!(Foo23 Foo22);
fooN!(Foo24 Foo23);
fooN!(Foo25 Foo24);
fooN!(Foo26 Foo25);
fooN!(Foo27 Foo26);

fn main() {
    let _foo = Foo27::Empty;
}
