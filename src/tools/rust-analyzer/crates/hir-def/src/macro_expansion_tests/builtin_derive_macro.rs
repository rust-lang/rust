//! Tests for `builtin_derive_macro.rs` from `hir_expand`.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn test_copy_expand_simple() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo;

impl < > core::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_in_core() {
    cov_mark::check!(test_copy_expand_in_core);
    check(
        r#"
//- /lib.rs crate:core
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;
"#,
        expect![[r#"
#[rustc_builtin_macro]
macro derive {}
#[rustc_builtin_macro]
macro Copy {}
#[derive(Copy)]
struct Foo;

impl < > crate ::marker::Copy for Foo< > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_type_params() {
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B>;

impl <A: core::marker::Copy, B: core::marker::Copy, > core::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_copy_expand_with_lifetimes() {
    // We currently just ignore lifetimes
    check(
        r#"
//- minicore: derive, copy
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;
"#,
        expect![[r#"
#[derive(Copy)]
struct Foo<A, B, 'a, 'b>;

impl <A: core::marker::Copy, B: core::marker::Copy, > core::marker::Copy for Foo<A, B, > where {}"#]],
    );
}

#[test]
fn test_clone_expand() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}

impl <A: core::clone::Clone, B: core::clone::Clone, > core::clone::Clone for Command<A, B, > where {
    fn clone(&self ) -> Self {
        match self {
            Command::Move {
                x: x, y: y,
            }
            =>Command::Move {
                x: x.clone(), y: y.clone(),
            }
            , Command::Do(f0, )=>Command::Do(f0.clone(), ), Command::Jump=>Command::Jump,
        }
    }
}"#]],
    );
}

#[test]
fn test_clone_expand_with_const_generics() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);
"#,
        expect![[r#"
#[derive(Clone)]
struct Foo<const X: usize, T>(u32);

impl <const X: usize, T: core::clone::Clone, > core::clone::Clone for Foo<X, T, > where {
    fn clone(&self ) -> Self {
        match self {
            Foo(f0, )=>Foo(f0.clone(), ),
        }
    }
}"#]],
    );
}

#[test]
fn test_default_expand() {
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
struct Foo {
    field1: i32,
    field2: (),
}
#[derive(Default)]
enum Bar {
    Foo(u8),
    #[default]
    Bar,
}
"#,
        expect![[r#"
#[derive(Default)]
struct Foo {
    field1: i32,
    field2: (),
}
#[derive(Default)]
enum Bar {
    Foo(u8),
    #[default]
    Bar,
}

impl < > core::default::Default for Foo< > where {
    fn default() -> Self {
        Foo {
            field1: core::default::Default::default(), field2: core::default::Default::default(),
        }
    }
}
impl < > core::default::Default for Bar< > where {
    fn default() -> Self {
        Bar::Bar
    }
}"#]],
    );
}

#[test]
fn test_partial_eq_expand() {
    check(
        r#"
//- minicore: derive, eq
#[derive(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(PartialEq, Eq)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl < > core::cmp::PartialEq for Command< > where {
    fn eq(&self , other: &Self ) -> bool {
        match (self , other) {
            (Command::Move {
                x: x_self, y: y_self,
            }
            , Command::Move {
                x: x_other, y: y_other,
            }
            )=>x_self.eq(x_other) && y_self.eq(y_other), (Command::Do(f0_self, ), Command::Do(f0_other, ))=>f0_self.eq(f0_other), (Command::Jump, Command::Jump)=>true , _unused=>false
        }
    }
}
impl < > core::cmp::Eq for Command< > where {}"#]],
    );
}

#[test]
fn test_partial_ord_expand() {
    check(
        r#"
//- minicore: derive, ord
#[derive(PartialOrd, Ord)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(PartialOrd, Ord)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl < > core::cmp::PartialOrd for Command< > where {
    fn partial_cmp(&self , other: &Self ) -> core::option::Option::Option<core::cmp::Ordering> {
        match core::intrinsics::discriminant_value(self ).partial_cmp(&core::intrinsics::discriminant_value(other)) {
            core::option::Option::Some(core::cmp::Ordering::Equal)=> {
                match (self , other) {
                    (Command::Move {
                        x: x_self, y: y_self,
                    }
                    , Command::Move {
                        x: x_other, y: y_other,
                    }
                    )=>match x_self.partial_cmp(&x_other) {
                        core::option::Option::Some(core::cmp::Ordering::Equal)=> {
                            match y_self.partial_cmp(&y_other) {
                                core::option::Option::Some(core::cmp::Ordering::Equal)=> {
                                    core::option::Option::Some(core::cmp::Ordering::Equal)
                                }
                                c=>return c,
                            }
                        }
                        c=>return c,
                    }
                    , (Command::Do(f0_self, ), Command::Do(f0_other, ))=>match f0_self.partial_cmp(&f0_other) {
                        core::option::Option::Some(core::cmp::Ordering::Equal)=> {
                            core::option::Option::Some(core::cmp::Ordering::Equal)
                        }
                        c=>return c,
                    }
                    , (Command::Jump, Command::Jump)=>core::option::Option::Some(core::cmp::Ordering::Equal), _unused=>core::option::Option::Some(core::cmp::Ordering::Equal)
                }
            }
            c=>return c,
        }
    }
}
impl < > core::cmp::Ord for Command< > where {
    fn cmp(&self , other: &Self ) -> core::cmp::Ordering {
        match core::intrinsics::discriminant_value(self ).cmp(&core::intrinsics::discriminant_value(other)) {
            core::cmp::Ordering::Equal=> {
                match (self , other) {
                    (Command::Move {
                        x: x_self, y: y_self,
                    }
                    , Command::Move {
                        x: x_other, y: y_other,
                    }
                    )=>match x_self.cmp(&x_other) {
                        core::cmp::Ordering::Equal=> {
                            match y_self.cmp(&y_other) {
                                core::cmp::Ordering::Equal=> {
                                    core::cmp::Ordering::Equal
                                }
                                c=>return c,
                            }
                        }
                        c=>return c,
                    }
                    , (Command::Do(f0_self, ), Command::Do(f0_other, ))=>match f0_self.cmp(&f0_other) {
                        core::cmp::Ordering::Equal=> {
                            core::cmp::Ordering::Equal
                        }
                        c=>return c,
                    }
                    , (Command::Jump, Command::Jump)=>core::cmp::Ordering::Equal, _unused=>core::cmp::Ordering::Equal
                }
            }
            c=>return c,
        }
    }
}"#]],
    );
}

#[test]
fn test_hash_expand() {
    check(
        r#"
//- minicore: derive, hash
use core::hash::Hash;

#[derive(Hash)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
use core::hash::Hash;

#[derive(Hash)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl < > core::hash::Hash for Command< > where {
    fn hash<H: core::hash::Hasher>(&self , state: &mut H) {
        core::mem::discriminant(self ).hash(state);
        match self {
            Command::Move {
                x: x, y: y,
            }
            => {
                x.hash(state);
                y.hash(state);
            }
            , Command::Do(f0, )=> {
                f0.hash(state);
            }
            , Command::Jump=> {}
            ,
        }
    }
}"#]],
    );
}

#[test]
fn test_debug_expand() {
    check(
        r#"
//- minicore: derive, fmt
use core::fmt::Debug;

#[derive(Debug)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
use core::fmt::Debug;

#[derive(Debug)]
enum Command {
    Move { x: i32, y: i32 },
    Do(&'static str),
    Jump,
}

impl < > core::fmt::Debug for Command< > where {
    fn fmt(&self , f: &mut core::fmt::Formatter) -> core::fmt::Result {
        match self {
            Command::Move {
                x: x, y: y,
            }
            =>f.debug_struct("Move").field("x", &x).field("y", &y).finish(), Command::Do(f0, )=>f.debug_tuple("Do").field(&f0).finish(), Command::Jump=>f.write_str("Jump"),
        }
    }
}"#]],
    );
}
