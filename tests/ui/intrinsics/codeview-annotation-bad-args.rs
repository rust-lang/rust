// Verifies that `codeview_annotation` emits a compile-time error
// when the `CodeViewAnnotationArgs::ARGS` associated constant
// does not const eval successfully.

//@ build-fail
//@ only-msvc

#![feature(codeview_annotation)]

use std::hint::{CodeViewAnnotationArgs, codeview_annotation};

// `CodeViewAnnotationArgs::ARGS` does not const eval successfully
struct Invalid;

impl CodeViewAnnotationArgs for Invalid {
    const ARGS: &[&str] = panic!("panic"); //~ ERROR evaluation panicked: panic [E0080]
}


// `CodeViewAnnotationArgs::ARGS` does not const eval successfully
// with generics involved
trait GetName {
    const NAME: &str;
}

impl GetName for i32 {
    const NAME: &str = panic!("panic"); //~ ERROR evaluation panicked: panic [E0080]
}

struct Args<T>(std::marker::PhantomData<T>);

impl<T: GetName> CodeViewAnnotationArgs for Args<T> {
    const ARGS: &[&str] = &[T::NAME];
}

fn emit<T: GetName>() {
    codeview_annotation::<Args<T>>();
}


fn main() {
    codeview_annotation::<Invalid>();
    emit::<i32>();
}
