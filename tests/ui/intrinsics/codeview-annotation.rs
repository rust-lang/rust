// Verifies that calls to the `codeview_annotation` API
// and the intrinsic compile successfully

//@ build-pass

#![feature(codeview_annotation)]
#![feature(core_intrinsics)]

// === Helper macros ===
macro_rules! call_intrinsic {
    ($args:expr) => {{
        call_codeview_annotation!(std::intrinsics, $args);
    }};
}

macro_rules! call_api {
    ($args:expr) => {{
        call_codeview_annotation!(std::hint, $args);
    }};
}

macro_rules! call_codeview_annotation {
    ($($module:ident)::+, $args:expr) => {{
        struct Args;

        impl std::hint::CodeViewAnnotationArgs for Args {
            const ARGS: &[&str] = $args;
        }

        $($module)::+::codeview_annotation::<Args>();
    }};
}

// === API tests ===
fn single() {
    call_api!(&["string1"]);
}

fn multiple() {
    call_api!(&["string1", "string2", "string3"]);
}

const STR_A: &str = "string1";
const STR_B: &str = "string2";
const STR_C: &str = "string3";

fn named_consts() {
    call_api!(&[STR_A, STR_B, STR_C]);
}

fn mixed_named_consts_and_literals() {
    call_api!(&[STR_A, "string2", "string3"]);
}

const STRS_SLICE: &[&str] = &["string1", "string2", "string3"];

fn named_const_slice() {
    call_api!(STRS_SLICE);
}

const STRS_ARRAY: [&str; 3] = ["string1", "string2", "string3"];

fn named_const_array_ref() {
    call_api!(&STRS_ARRAY);
}

static STATIC_STRS_ARRAY: [&str; 3] = ["string1", "string2", "string3"];

fn static_array_ref() {
    call_api!(&STATIC_STRS_ARRAY);
}

// Data associated with the types of
// some variables passed to codeview_annotation
// e.g. type names
fn consts_associated_with_vars() {
    // A trait that lets you assign names to types
    trait TypeName {
        const NAME: &str;
    }

    struct A;
    impl TypeName for A {
        const NAME: &str = "A";
    }

    struct B;
    impl TypeName for B {
        const NAME: &str = "B";
    }

    // This function's purpose is to monomorphize and infer
    // the types of its arguments and pass their associated
    // consts to codeview_annotation
    fn emit_annotation<T1: TypeName, T2: TypeName>(_var1: &T1, _var2: &T2) {
        struct Args<T1, T2>(std::marker::PhantomData<(T1, T2)>);

        impl<T1: TypeName, T2: TypeName> std::hint::CodeViewAnnotationArgs for Args<T1, T2> {
            const ARGS: &[&str] = &["string", T1::NAME, T2::NAME];
        }

        std::hint::codeview_annotation::<Args<T1, T2>>();
    }

    // Strings "A" and "B" get passed to
    // codeview_annotation for a and b respectively
    let a = A;
    let b = B;
    emit_annotation(&a, &b);
}

// An associated const slice passed as args
// to codeview_annotation
trait HasStrs {
    const STRS: &[&str];
}

impl HasStrs for i32 {
    const STRS: &[&str] = &["string1", "string2", "string3"];
}

fn generic_associated_const_slice<T: HasStrs>() {
    struct Args<T>(std::marker::PhantomData<T>);

    impl<T: HasStrs> std::hint::CodeViewAnnotationArgs for Args<T> {
        const ARGS: &[&str] = T::STRS;
    }

    std::hint::codeview_annotation::<Args<T>>();
}

fn empty_strings() {
    call_api!(&["", "", "string1"]);
}

fn empty_slice() {
    call_api!(&[]);
}

const EMPTY_STRS_SLICE: &[&str] = &[];

pub fn named_empty_slice() {
    call_api!(EMPTY_STRS_SLICE);
}

// === Intrinsic tests ===
fn intrinsic_single() {
    call_intrinsic!(&["string1"]);
}

fn intrinsic_multiple() {
    call_intrinsic!(&["string1", "string2", "string3"]);
}

fn intrinsic_mixed_named_consts_and_literals() {
    call_intrinsic!(&[STR_A, "string2", "string3"]);
}

fn main() {
    single();
    multiple();
    named_consts();
    mixed_named_consts_and_literals();
    named_const_slice();
    named_const_array_ref();
    static_array_ref();
    consts_associated_with_vars();
    generic_associated_const_slice::<i32>();
    empty_strings();
    empty_slice();
    named_empty_slice();

    intrinsic_single();
    intrinsic_multiple();
    intrinsic_mixed_named_consts_and_literals();
}
