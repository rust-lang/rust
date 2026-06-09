//@ check-pass

type _TaWhere1<T> where T: Iterator<Item: Copy> = T; //~ WARNING type_alias_bounds
type _TaWhere2<T> where T: Iterator<Item: 'static> = T; //~ WARNING type_alias_bounds
type _TaWhere3<T> where T: Iterator<Item: 'static> = T; //~ WARNING type_alias_bounds
type _TaWhere4<T> where T: Iterator<Item: 'static + Copy + Send> = T; //~ WARNING type_alias_bounds
type _TaWhere5<T> where T: Iterator<Item: for<'a> Into<&'a u8>> = T; //~ WARNING type_alias_bounds
type _TaWhere6<T> where T: Iterator<Item: Iterator<Item: Copy>> = T; //~ WARNING type_alias_bounds

type _TaInline1<T: Iterator<Item: Copy>> = T; //~ WARNING type_alias_bounds
type _TaInline2<T: Iterator<Item: 'static>> = T; //~ WARNING type_alias_bounds
type _TaInline3<T: Iterator<Item: 'static>> = T; //~ WARNING type_alias_bounds
type _TaInline4<T: Iterator<Item: 'static + Copy + Send>> = T; //~ WARNING type_alias_bounds
type _TaInline5<T: Iterator<Item: for<'a> Into<&'a u8>>> = T; //~ WARNING type_alias_bounds
type _TaInline6<T: Iterator<Item: Iterator<Item: Copy>>> = T; //~ WARNING type_alias_bounds

fn main() {}
