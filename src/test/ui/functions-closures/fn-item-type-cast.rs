// run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// Test explicit coercions from a fn item type to a fn pointer type.


fn foo(x: isize) -> isize { x * 2 }
fn bar(x: isize) -> isize { x * 4 }
type IntMap = fn(isize) -> isize;

fn eq<T>(x: T, y: T) { }

static TEST: Option<IntMap> = Some(foo as IntMap);

fn main() {
    let f = foo as IntMap;

    let f = if true { foo as IntMap } else { bar as IntMap };
    assert_eq!(f(4), 8);

    eq(foo as IntMap, bar as IntMap);
}
