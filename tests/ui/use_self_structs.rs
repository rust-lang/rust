#![warn(clippy::use_self)]
#![allow(clippy::type_complexity)]

fn main() {}

struct Basic {
    flag: Option<Box<Basic>>,
    //~^ use_self
}

struct BasicSelf {
    okay: Option<Box<Self>>,
}

struct Generic<'q, T: From<u8>> {
    t: &'q T,
    flag: Option<Box<Generic<'q, T>>>,
    //~^ use_self
}

struct GenericSelf<'q, T: From<u8>> {
    t: &'q T,
    okay: Option<Box<Self>>,
}

struct MixedLifetimes<'q, T: From<u8> + 'static> {
    t: &'q T,
    okay: Option<Box<MixedLifetimes<'static, T>>>,
}

struct ConcreteType<'q, T: From<u8>> {
    t: &'q T,
    okay: Option<Box<ConcreteType<'q, u64>>>,
}

struct ConcreteAndGeneric<'q, T: From<u8>> {
    t: &'q T,
    flag: Option<Box<ConcreteAndGeneric<'q, T>>>,
    //~^ use_self
    okay: Option<Box<ConcreteAndGeneric<'q, u64>>>,
}

struct ConcreteAndGenericSelf<'q, T: From<u8>> {
    t: &'q T,
    okay_1: Option<Box<Self>>,
    okay_2: Option<Box<ConcreteAndGeneric<'q, u64>>>,
}

macro_rules! recursive_struct {
    ($name:ident) => {
        struct $name {
            okay: Option<Box<$name>>,
        }
    };
}

recursive_struct!(X);
recursive_struct!(Y);
recursive_struct!(Z);

struct Tree {
    left: Option<Box<Tree>>,
    //~^ use_self
    right: Option<Box<Tree>>,
    //~^ use_self
}

struct TreeSelf {
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

struct TreeMixed {
    left: Option<Box<Self>>,
    right: Option<Box<TreeMixed>>,
    //~^ use_self
}

struct Nested {
    flag: Option<Box<Option<Box<Nested>>>>,
    //~^ use_self
}

struct NestedSelf {
    okay: Option<Box<Option<Box<Self>>>>,
}

struct Tuple(Option<Box<Tuple>>);
//~^ use_self

struct TupleSelf(Option<Box<Self>>);

use std::cell::RefCell;
use std::rc::{Rc, Weak};

struct Containers {
    flag: Vec<Option<Rc<RefCell<Weak<Vec<Box<Containers>>>>>>>,
    //~^ use_self
}

struct ContainersSelf {
    okay: Vec<Option<Rc<RefCell<Weak<Vec<Box<Self>>>>>>>,
}

type Wrappers<T> = Vec<Option<Rc<RefCell<Weak<Vec<Box<T>>>>>>>;

struct Alias {
    flag: Wrappers<Alias>,
    //~^ use_self
}

struct AliasSelf {
    okay: Wrappers<Self>,
}

struct Array<const N: usize> {
    flag: [Option<Box<Array<N>>>; N],
    //~^ use_self
}

struct ArraySelf<const N: usize> {
    okay: [Option<Box<Self>>; N],
}

enum Enum {
    Nil,
    Cons(Box<Enum>),
    //~^ use_self
}

enum EnumSelf {
    Nil,
    Cons(Box<Self>),
}
