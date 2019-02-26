// We need all these 9 issue-20616-N.rs files
// because we can only catch one parsing error at a time

type Type_1_<'a, T> = &'a T;


//type Type_1<'a T> = &'a T; // error: expected `,` or `>` after lifetime name, found `T`


//type Type_2 = Type_1_<'static ()>; // error: expected `,` or `>` after lifetime name, found `(`


//type Type_3<T> = Box<T,,>; // error: expected type, found `,`


//type Type_4<T> = Type_1_<'static,, T>; // error: expected type, found `,`


type Type_5_<'a> = Type_1_<'a, ()>;


//type Type_5<'a> = Type_1_<'a, (),,>; // error: expected type, found `,`


//type Type_6 = Type_5_<'a,,>; // error: expected type, found `,`


//type Type_7 = Box<(),,>; // error: expected type, found `,`


//type Type_8<'a,,> = &'a (); // error: expected identifier, found `,`


type Type_9<T,,> = Box<T>;
//~^ error: expected one of `>`, `const`, identifier, or lifetime, found `,`
