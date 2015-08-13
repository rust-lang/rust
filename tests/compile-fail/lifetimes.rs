#![feature(plugin)]
#![plugin(clippy)]

#![deny(needless_lifetimes)]

fn distinct_lifetimes<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: u8) { }
//~^ERROR explicit lifetimes given

fn distinct_and_static<'a, 'b>(_x: &'a u8, _y: &'b u8, _z: &'static u8) { }
//~^ERROR explicit lifetimes given

fn same_lifetime_on_input<'a>(_x: &'a u8, _y: &'a u8) { } // no error, same lifetime on two params

fn only_static_on_input(_x: &u8, _y: &u8, _z: &'static u8) { } // no error, static involved

fn in_and_out<'a>(x: &'a u8, _y: u8) -> &'a u8 { x }
//~^ERROR explicit lifetimes given

fn multiple_in_and_out_1<'a>(x: &'a u8, _y: &'a u8) -> &'a u8 { x } // no error, multiple input refs

fn multiple_in_and_out_2<'a, 'b>(x: &'a u8, _y: &'b u8) -> &'a u8 { x } // no error, multiple input refs

fn in_static_and_out<'a>(x: &'a u8, _y: &'static u8) -> &'a u8 { x } // no error, static involved

fn deep_reference_1<'a, 'b>(x: &'a u8, _y: &'b u8) -> Result<&'a u8, ()> { Ok(x) } // no error

fn deep_reference_2<'a>(x: Result<&'a u8, &'a u8>) -> &'a u8 { x.unwrap() } // no error, two input refs

fn deep_reference_3<'a>(x: &'a u8, _y: u8) -> Result<&'a u8, ()> { Ok(x) }
//~^ERROR explicit lifetimes given

type Ref<'r> = &'r u8;

fn lifetime_param_1<'a>(_x: Ref<'a>, _y: &'a u8) { } // no error, same lifetime on two params

fn lifetime_param_2<'a, 'b>(_x: Ref<'a>, _y: &'b u8) { }
//~^ERROR explicit lifetimes given

fn lifetime_param_3<'a, 'b: 'a>(_x: Ref<'a>, _y: &'b u8) { } // no error, bounded lifetime

struct X {
    x: u8,
}

impl X {
    fn self_and_out<'s>(&'s self) -> &'s u8 { &self.x }
    //~^ERROR explicit lifetimes given

    fn self_and_in_out<'s, 't>(&'s self, _x: &'t u8) -> &'s u8 { &self.x } // no error, multiple input refs

    fn distinct_self_and_in<'s, 't>(&'s self, _x: &'t u8) { }
    //~^ERROR explicit lifetimes given

    fn self_and_same_in<'s>(&'s self, _x: &'s u8) { } // no error, same lifetimes on two params
}

static STATIC: u8 = 1;

fn main() {
    distinct_lifetimes(&1, &2, 3);
    distinct_and_static(&1, &2, &STATIC);
    same_lifetime_on_input(&1, &2);
    only_static_on_input(&1, &2, &STATIC);
    in_and_out(&1, 2);
    multiple_in_and_out_1(&1, &2);
    multiple_in_and_out_2(&1, &2);
    in_static_and_out(&1, &STATIC);
    let _ = deep_reference_1(&1, &2);
    let _ = deep_reference_2(Ok(&1));
    let _ = deep_reference_3(&1, 2);
    lifetime_param_1(&1, &2);
    lifetime_param_2(&1, &2);
    lifetime_param_3(&1, &2);

    let foo = X { x: 1 };
    foo.self_and_out();
    foo.self_and_in_out(&1);
    foo.distinct_self_and_in(&1);
    foo.self_and_same_in(&1);
}
