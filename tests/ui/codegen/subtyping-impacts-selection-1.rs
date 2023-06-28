// run-pass
// revisions: mir codegen
//[mir] compile-flags: -Zmir-opt-level=3
//[codegen] compile-flags: -Zmir-opt-level=0

// A regression test for #107205
#![allow(coherence_leak_check)]
struct Foo<T: 'static>(T);

fn useful<'a>(_: &'a u8) {}

trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}

impl GetInner for Foo<fn(&'static u8)> {
    type Assoc = String;
    fn muahaha(&mut self) -> String {
        String::from("I am a string")
    }
}

impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] {
        [100; 3]
    }
}

fn break_me(hr_fnptr: Box<Foo::<for<'a> fn(&'a u8)>>) -> Box<dyn GetInner<Assoc = String>> {
    let lr_fnptr = hr_fnptr as Box<Foo<fn(&'static u8)>>;
    lr_fnptr as Box<dyn GetInner<Assoc = String>>
}

fn main() {
    drop(Box::new(Foo(useful as fn(&'static u8))) as Box<dyn GetInner<Assoc = String>>);
    drop(Box::new(Foo(useful as fn(&u8))) as Box<dyn GetInner<Assoc = [usize; 3]>>);

    let mut any = break_me(Box::new(Foo(useful)));

    let evil_string = any.muahaha();
    assert_eq!(evil_string, "I am a string");
}
