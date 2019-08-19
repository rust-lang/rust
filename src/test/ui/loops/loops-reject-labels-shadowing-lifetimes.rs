// Issue #21633: reject duplicate loop labels in function bodies.
// This is testing interaction between lifetime-params and labels.

// build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code, unused_variables)]

fn foo() {
    fn foo<'a>() {
        'a: loop { break 'a; }
        //~^ WARN label name `'a` shadows a lifetime name that is already in scope
    }

    struct Struct<'b, 'c> { _f: &'b i8, _g: &'c i8 }
    enum Enum<'d, 'e> { A(&'d i8), B(&'e i8) }

    impl<'d, 'e> Struct<'d, 'e> {
        fn meth_okay() {
            'a: loop { break 'a; }
            'b: loop { break 'b; }
            'c: loop { break 'c; }
        }
    }

    impl <'d, 'e> Enum<'d, 'e> {
        fn meth_okay() {
            'a: loop { break 'a; }
            'b: loop { break 'b; }
            'c: loop { break 'c; }
        }
    }

    impl<'bad, 'c> Struct<'bad, 'c> {
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }

    impl<'b, 'bad> Struct<'b, 'bad> {
        fn meth_bad2(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }

    impl<'b, 'c> Struct<'b, 'c> {
        fn meth_bad3<'bad>(x: &'bad i8) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }

        fn meth_bad4<'a,'bad>(x: &'a i8, y: &'bad i8) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }

    impl <'bad, 'e> Enum<'bad, 'e> {
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }
    impl <'d, 'bad> Enum<'d, 'bad> {
        fn meth_bad2(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }
    impl <'d, 'e> Enum<'d, 'e> {
        fn meth_bad3<'bad>(x: &'bad i8) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }

        fn meth_bad4<'a,'bad>(x: &'bad i8) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }

    trait HasDefaultMethod1<'bad> {
        fn meth_okay() {
            'c: loop { break 'c; }
        }
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }
    trait HasDefaultMethod2<'a,'bad> {
        fn meth_bad(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }
    trait HasDefaultMethod3<'a,'b> {
        fn meth_bad<'bad>(&self) {
            'bad: loop { break 'bad; }
            //~^ WARN label name `'bad` shadows a lifetime name that is already in scope
        }
    }
}


pub fn main() {
    foo();
}
