#![warn(clippy::pointer_format)]

use core::fmt::Debug;
use core::marker::PhantomData;

#[derive(Debug)]
struct ContainsPointerDeep {
    w: WithPointer,
}

struct ManualDebug {
    ptr: *const u8,
}

#[derive(Debug)]
struct WithPointer {
    len: usize,
    ptr: *const u8,
}

impl std::fmt::Debug for ManualDebug {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("ManualDebug")
    }
}

trait Foo {
    type Assoc: Foo + Debug;
}

#[derive(Debug)]
struct S<T: Foo + 'static>(&'static S<T::Assoc>, PhantomData<T>);

#[allow(unused)]
fn unbounded<T: Foo + Debug + 'static>(s: &S<T>) {
    format!("{s:?}");
}

fn main() {
    let m = &(main as fn());
    let g = &0;
    let o = &format!("{m:p}");
    //~^ pointer_format
    let _ = format!("{m:?}");
    //~^ pointer_format
    println!("{g:p}");
    //~^ pointer_format
    panic!("{o:p}");
    //~^ pointer_format
    let answer = 42;
    let x = &raw const answer;
    let arr = [0u8; 8];
    let with_ptr = WithPointer { len: 8, ptr: &arr as _ };
    let _ = format!("{x:?}");
    //~^ pointer_format
    print!("{with_ptr:?}");
    //~^ pointer_format
    let container = ContainsPointerDeep { w: with_ptr };
    print!("{container:?}");
    //~^ pointer_format

    let no_pointer = "foo";
    println!("{no_pointer:?}");
    let manual_debug = ManualDebug { ptr: &arr as _ };
    println!("{manual_debug:?}");
}
