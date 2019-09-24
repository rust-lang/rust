// build-pass (FIXME(62277): could be check-pass?)

#![feature(associated_type_bounds)]

type _TaWhere1<T> where T: Iterator<Item: Copy> = T;
type _TaWhere2<T> where T: Iterator<Item: 'static> = T;
type _TaWhere3<T> where T: Iterator<Item: 'static> = T;
type _TaWhere4<T> where T: Iterator<Item: 'static + Copy + Send> = T;
type _TaWhere5<T> where T: Iterator<Item: for<'a> Into<&'a u8>> = T;
type _TaWhere6<T> where T: Iterator<Item: Iterator<Item: Copy>> = T;

type _TaInline1<T: Iterator<Item: Copy>> = T;
type _TaInline2<T: Iterator<Item: 'static>> = T;
type _TaInline3<T: Iterator<Item: 'static>> = T;
type _TaInline4<T: Iterator<Item: 'static + Copy + Send>> = T;
type _TaInline5<T: Iterator<Item: for<'a> Into<&'a u8>>> = T;
type _TaInline6<T: Iterator<Item: Iterator<Item: Copy>>> = T;

fn main() {}
