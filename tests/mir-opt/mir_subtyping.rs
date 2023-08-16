// compile-flags: -Z mir-opt-level=0

// EMIT_MIR mir_subtyping.main.Subtyper.diff
#![allow(coherence_leak_check)]

struct Foo<T: 'static>(T);

fn useful<'a>(_: &'a u8) {}

pub struct Wrapper(for<'b> fn(&'b u8));

trait GetInner {
    type Assoc;
    fn muahaha(&mut self) -> Self::Assoc;
}

impl GetInner for Foo<fn(&'static u8)> {
    type Assoc = String;
    fn muahaha(&mut self) -> String {
        panic!("cant do it boss")
    }
}

impl GetInner for Foo<for<'a> fn(&'a u8)> {
    type Assoc = [usize; 3];
    fn muahaha(&mut self) -> [usize; 3] {
        [100; 3]
    }
}

fn main() {
    let wrapper = Wrapper(useful);

    drop(Box::new(Foo(useful as fn(&'static u8))) as Box<dyn GetInner<Assoc = String>>);
    drop(Box::new(Foo(useful as fn(&u8))) as Box<dyn GetInner<Assoc = [usize; 3]>>);

    let hr_fnptr = Box::new(Foo::<for<'a> fn(&'a u8)>(wrapper.0));
    let lr_fnptr = hr_fnptr as Box<Foo<fn(&'static u8)>>;
    let mut any = lr_fnptr as Box<dyn GetInner<Assoc = String>>;

    let evil_string = any.muahaha();
    drop(evil_string);
}
