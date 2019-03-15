#[prelude_import]
use ::std::prelude::v1::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:hir,typed
// pp-exact:issue-4264.pp

// #4264 fixed-length vector types

pub fn foo(_: [i32; (3 as usize)]) ({ } as ())

pub fn bar() ({
                  const FOO: usize = ((5 as usize) - (4 as usize) as usize);
                  let _: [(); (FOO as usize)] = ([(() as ())] as [(); 1]);

                  let _: [(); (1 as usize)] = ([(() as ())] as [(); 1]);

                  let _ =
                      (((&([(1 as i32), (2 as i32), (3 as i32)] as [i32; 3])
                            as &[i32; 3]) as *const _ as *const [i32; 3]) as
                          *const [i32; (3 as usize)] as *const [i32; 3]);









                  (($crate::fmt::format as
                       for<'r> fn(std::fmt::Arguments<'r>) -> std::string::String {std::fmt::format})(((<$crate::fmt::Arguments>::new_v1
                                                                                                           as
                                                                                                           fn(&[&str], &[std::fmt::ArgumentV1<'_>]) -> std::fmt::Arguments<'_> {std::fmt::Arguments::<'_>::new_v1})((&([("test"
                                                                                                                                                                                                                            as
                                                                                                                                                                                                                            &'static str)]
                                                                                                                                                                                                                          as
                                                                                                                                                                                                                          [&str; 1])
                                                                                                                                                                                                                        as
                                                                                                                                                                                                                        &[&str; 1]),
                                                                                                                                                                                                                    (&(match (()
                                                                                                                                                                                                                                 as
                                                                                                                                                                                                                                 ())
                                                                                                                                                                                                                           {
                                                                                                                                                                                                                           ()
                                                                                                                                                                                                                           =>
                                                                                                                                                                                                                           ([]
                                                                                                                                                                                                                               as
                                                                                                                                                                                                                               [std::fmt::ArgumentV1<'_>; 0]),
                                                                                                                                                                                                                       }
                                                                                                                                                                                                                          as
                                                                                                                                                                                                                          [std::fmt::ArgumentV1<'_>; 0])
                                                                                                                                                                                                                        as
                                                                                                                                                                                                                        &[std::fmt::ArgumentV1<'_>; 0]))
                                                                                                          as
                                                                                                          std::fmt::Arguments<'_>))
                      as std::string::String);
              } as ())
pub type Foo = [i32; (3 as usize)];
pub struct Bar {
    pub x: [i32; (3 as usize)],
}
pub struct TupleBar([i32; (4 as usize)]);
pub enum Baz { BazVariant([i32; (5 as usize)]), }
pub fn id<T>(x: T) -> T ({ (x as T) } as T)
pub fn use_id() ({
                     let _ =
                         ((id::<[i32; (3 as usize)]> as
                              fn([i32; 3]) -> [i32; 3] {id::<[i32; 3]>})(([(1
                                                                               as
                                                                               i32),
                                                                           (2
                                                                               as
                                                                               i32),
                                                                           (3
                                                                               as
                                                                               i32)]
                                                                             as
                                                                             [i32; 3]))
                             as [i32; 3]);
                 } as ())
fn main() ({ } as ())
