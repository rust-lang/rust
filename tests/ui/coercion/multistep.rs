//@ check-fail
//@ known-bug: #148283
//@ failure-status: 101
//@ rustc-env:RUST_BACKTRACE=0

#![allow(static_mut_refs)]
#![allow(dead_code)]
use std::ops::Deref;

pub static mut ACTIONS: Vec<&'static str> = Vec::new();

pub struct Wrap<T: ?Sized>(T);

// Deref Chain: FinalType <- UnsizedArray <- IntWrapper <- ArrayWrapper <- TopType
pub struct TopType;
pub type ArrayWrapper = Wrap<[i32; 0]>;
pub struct IntWrapper;
pub type UnsizedArray = Wrap<[i32]>;
pub struct FinalType;
pub struct TopTypeNoTrait;

impl Deref for TopType {
    type Target = ArrayWrapper;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref TopType->ArrayWrapper"); }
        &Wrap([])
    }
}

impl Deref for ArrayWrapper {
    type Target = IntWrapper;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref ArrayWrapper->IntWrapper"); }
        &IntWrapper
    }
}

impl Deref for IntWrapper {
    type Target = UnsizedArray;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref IntWrapper->UnsizedArray"); }
        &Wrap([])
    }
}

impl Deref for UnsizedArray {
    type Target = FinalType;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref UnsizedArray->FinalType"); }
        &FinalType
    }
}

impl Deref for TopTypeNoTrait {
    type Target = ArrayWrapper;
    fn deref(&self) -> &Self::Target {
        unsafe { ACTIONS.push("deref TopTypeNoTrait->ArrayWrapper"); }
        &Wrap([])
    }
}

trait Trait {
    fn self_ty(&self);

    fn complete(&self) -> Vec<&'static str> {
        self.self_ty();
        let actions = unsafe { ACTIONS.clone() };
        unsafe { ACTIONS.clear() };
        actions
    }
}

impl Trait for TopType {
    fn self_ty(&self) {
        unsafe { ACTIONS.push("self_ty TopType"); }
    }
}

impl Trait for ArrayWrapper {
    fn self_ty(&self) {
        unsafe { ACTIONS.push("self_ty ArrayWrapper"); }
    }
}

impl Trait for IntWrapper {
    fn self_ty(&self) {
        unsafe { ACTIONS.push("self_ty IntWrapper"); }
    }
}

impl Trait for UnsizedArray {
    fn self_ty(&self) {
        unsafe { ACTIONS.push("self_ty UnsizedArray"); }
    }
}

impl Trait for FinalType {
    fn self_ty(&self) {
        unsafe { ACTIONS.push("self_ty FinalType"); }
    }
}

fn simple() {
    let x = match 0 {
        0 => &Wrap([]) as &ArrayWrapper,
        _ => &Wrap([]) as &UnsizedArray,
    };
    assert_eq!(x.complete(), vec!["self_ty UnsizedArray"]);
}

fn long_chain() {
    let x = match 0 {
        0 => &TopType      as &TopType,
        1 => &Wrap([])     as &ArrayWrapper,
        2 => &IntWrapper   as &IntWrapper,
        3 => &Wrap([])     as &UnsizedArray,
        4 => &FinalType    as &FinalType,
        _ => loop {},
    };
    assert_eq!(
        x.complete(),
        vec![
            "deref TopType->ArrayWrapper",
            "deref ArrayWrapper->IntWrapper",
            "deref IntWrapper->UnsizedArray",
            "deref UnsizedArray->FinalType",
            "self_ty FinalType",
        ],
    );
}

fn mixed_coercion() {
    let x = match 0 {
        0 => &TopType      as &TopType,
        1 => &Wrap([])     as &ArrayWrapper,
        // IntWrapper arm removed
        2 => &Wrap([])     as &UnsizedArray,
        3 => &FinalType    as &FinalType,
        _ => loop {},
    };
    assert_eq!(
        x.complete(),
        vec![
            "deref TopType->ArrayWrapper",
            "deref UnsizedArray->FinalType",
            "self_ty FinalType",
        ]
    );
}

fn order_dependence() {
    let a = match 0 {
        0 => &Wrap([])   as &ArrayWrapper,
        1 => &IntWrapper as &IntWrapper,
        2 => &Wrap([])   as &UnsizedArray,
        _ => loop {},
    };
    assert_eq!(
        a.complete(),
        vec![
            "deref ArrayWrapper->IntWrapper",
            "deref IntWrapper->UnsizedArray",
            "self_ty UnsizedArray",
        ],
    );

    unsafe { ACTIONS.clear() }
    let b = match 0 {
        0 => &Wrap([])   as &ArrayWrapper,
        1 => &Wrap([])   as &UnsizedArray,
        2 => &IntWrapper as &IntWrapper,
        _ => loop {},
    };
    assert_eq!(b.complete(), vec!["self_ty UnsizedArray"]);
}

fn deref_to_dyn() {
    let x = match 0 {
        0 => &TopTypeNoTrait as &TopTypeNoTrait,
        1 => &TopTypeNoTrait as &FinalType,
        2 => &TopTypeNoTrait as &FinalType as &dyn Trait,
        _ => loop {},
    };
    assert_eq!(
        x.complete(),
        vec![
            "deref TopTypeNoTrait->ArrayWrapper",
            "deref ArrayWrapper->IntWrapper",
            "deref IntWrapper->UnsizedArray",
            "deref UnsizedArray->FinalType",
            "self_ty FinalType",
        ],
    );
}

fn main() {
    simple();
    long_chain();
    mixed_coercion();
    order_dependence();
    deref_to_dyn();
}
