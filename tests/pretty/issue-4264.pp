#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
// pretty-compare-only
// pretty-mode:hir,typed
// pp-exact:issue-4264.pp

// #4264 fixed-length vector types

fn foo(_: [i32; (3 as usize)]) ({ } as ())

fn bar() ({
        const FOO: usize = ((5 as usize) - (4 as usize) as usize);
        let _: [(); (FOO as usize)] = ([(() as ())] as [(); 1]);

        let _: [(); (1 as usize)] = ([(() as ())] as [(); 1]);

        let _ =
            (((&([(1 as i32), (2 as i32), (3 as i32)] as [i32; 3]) as
                        &[i32; 3]) as *const _ as *const [i32; 3]) as
                *const [i32; (3 as usize)] as *const [i32; 3]);









        ({
                let res =
                    ((::alloc::fmt::format as
                            for<'a> fn(Arguments<'a>) -> String {format})(((<#[lang = "format_arguments"]>::new_v1
                                as
                                fn(&[&'static str], &[core::fmt::ArgumentV1<'_>]) -> Arguments<'_> {Arguments::<'_>::new_v1})((&([("test"
                                            as &str)] as [&str; 1]) as &[&str; 1]),
                            (&([] as [core::fmt::ArgumentV1<'_>; 0]) as
                                &[core::fmt::ArgumentV1<'_>; 0])) as Arguments<'_>)) as
                        String);
                (res as String)
            } as String);
    } as ())
type Foo = [i32; (3 as usize)];
struct Bar {
    x: [i32; (3 as usize)],
}
struct TupleBar([i32; (4 as usize)]);
enum Baz { BazVariant([i32; (5 as usize)]), }
fn id<T>(x: T) -> T ({ (x as T) } as T)
fn use_id() ({
        let _ =
            ((id::<[i32; (3 as usize)]> as
                    fn([i32; 3]) -> [i32; 3] {id::<[i32; 3]>})(([(1 as i32),
                        (2 as i32), (3 as i32)] as [i32; 3])) as [i32; 3]);
    } as ())
fn main() ({ } as ())
