#![feature(plugin)]
#![plugin(clippy)]

#![deny(double_parens)]
#![allow(dead_code)]

fn dummy_fn<T>(_: T) {}

struct DummyStruct;

impl DummyStruct {
    fn dummy_method<T>(self, _: T) {}
}

fn simple_double_parens() -> i32 {
    ((0)) //~ERROR Consider removing unnecessary double parentheses
}

fn fn_double_parens() {
    dummy_fn((0)); //~ERROR Consider removing unnecessary double parentheses
}

fn method_double_parens(x: DummyStruct) {
    x.dummy_method((0)); //~ERROR Consider removing unnecessary double parentheses
}

fn tuple_double_parens() -> (i32, i32) {
    ((1, 2)) //~ERROR Consider removing unnecessary double parentheses
}

fn unit_double_parens() {
    (()) //~ERROR Consider removing unnecessary double parentheses
}

fn fn_tuple_ok() {
    dummy_fn((1, 2));
}

fn method_tuple_ok(x: DummyStruct) {
    x.dummy_method((1, 2));
}

fn fn_unit_ok() {
    dummy_fn(());
}

fn method_unit_ok(x: DummyStruct) {
    x.dummy_method(());
}

fn main() {}
