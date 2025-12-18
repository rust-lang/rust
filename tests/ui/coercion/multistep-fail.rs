//@ check-fail
//@ known-bug: #148283

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

fn deref_to_dyn_direct() {
    let x = match 0 {
        0 => &TopTypeNoTrait as &TopTypeNoTrait,
        1 => &TopTypeNoTrait as &FinalType as &dyn Trait,
        _ => loop {},
    };
}

fn main() {
    deref_to_dyn_direct();
}
