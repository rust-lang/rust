// check-pass

use std::cell::RefMut;

fn main() {
    StateMachine2::Init.resume();
}

enum StateMachine2<'a> {
    Init,
    #[allow(dead_code)] // match required for ICE
    AfterTwoYields {
        p: Backed<'a, *mut String>,
    },
}

impl<'a> StateMachine2<'a> {
    fn take(&self) -> Self {
        StateMachine2::Init
    }
}

impl<'a> StateMachine2<'a> {
    fn resume(&mut self) -> () {
        use StateMachine2::*;
        match self.take() {
            AfterTwoYields { p } => {
                p.with(|_| {});
            }
            _ => panic!("Resume after completed."),
        }
    }
}

unsafe trait Unpack<'a> {
    type Unpacked: 'a;

    fn unpack(&self) -> Self::Unpacked {
        unsafe { std::mem::transmute_copy(&self) }
    }
}

unsafe trait Pack {
    type Packed;

    fn pack(&self) -> Self::Packed {
        unsafe { std::mem::transmute_copy(&self) }
    }
}

unsafe impl<'a> Unpack<'a> for String {
    type Unpacked = String;
}

unsafe impl Pack for String {
    type Packed = String;
}

unsafe impl<'a> Unpack<'a> for *mut String {
    type Unpacked = &'a mut String;
}

unsafe impl<'a> Pack for &'a mut String {
    type Packed = *mut String;
}

struct Backed<'a, U>(RefMut<'a, Option<String>>, U);

impl<'a, 'b, U: Unpack<'b>> Backed<'a, U> {
    fn with<F>(self, f: F) -> Backed<'a, ()>
    where
        F: for<'f> FnOnce(<U as Unpack<'f>>::Unpacked) -> (),
    {
        let result: () = f(self.1.unpack());
        Backed(self.0, result)
    }
}
