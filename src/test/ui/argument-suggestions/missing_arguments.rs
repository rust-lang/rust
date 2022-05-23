fn one_arg(_a: i32) {}
fn two_same(_a: i32, _b: i32) {}
fn two_diff(_a: i32, _b: f32) {}
fn three_same(_a: i32, _b: i32, _c: i32) {}
fn three_diff(_a: i32, _b: f32, _c: &str) {}
fn four_repeated(_a: i32, _b: f32, _c: f32, _d: &str) {}
fn complex(_a: i32, _b: f32, _c: i32, _d: f32, _e: &str) {}

fn main() {
  one_arg(); //~ ERROR this function takes
  // The headers here show the types expected,
  // with formatting to emphasize which arguments are missing
  /*         i32     f32    */
  two_same(               ); //~ ERROR this function takes
  two_same(   1           ); //~ ERROR this function takes
  two_diff(               ); //~ ERROR this function takes
  two_diff(   1           ); //~ ERROR this function takes
  two_diff(          1.0  ); //~ ERROR this function takes

  /*           i32     i32     i32    */
  three_same(                       ); //~ ERROR this function takes
  three_same(   1                   ); //~ ERROR this function takes
  three_same(   1,      1           ); //~ ERROR this function takes

  /*           i32     f32     &str   */
  three_diff(          1.0,     ""  ); //~ ERROR this function takes
  three_diff(   1,              ""  ); //~ ERROR this function takes
  three_diff(   1,     1.0          ); //~ ERROR this function takes
  three_diff(                   ""  ); //~ ERROR this function takes
  three_diff(          1.0          ); //~ ERROR this function takes
  three_diff(   1                   ); //~ ERROR this function takes

  /*              i32     f32     f32     &str   */
  four_repeated(                               ); //~ ERROR this function takes
  four_repeated(   1,                     ""   ); //~ ERROR this function takes

  /*        i32   f32   i32   f32   &str   */
  complex(                               ); //~ ERROR this function takes
  complex(   1,                     ""   ); //~ ERROR this function takes
}
