type A = for<'a> &'a u32;
type B = for<'a> (&'a u32,);
type B = for<'a> [u32];
