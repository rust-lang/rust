// xfail-test
fn main()
{
// See #2969 -- error message should be improved
   let mut x = [mut 1, 2, 4];
   let v : &int = &x[2];
   x[2] = 6;
   assert *v == 6;
}
