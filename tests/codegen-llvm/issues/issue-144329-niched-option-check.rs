//! Ensure that redundant null checks on `&mut T` from `Option<(_, &mut T)>` are eliminated.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

type T = [u64; 4];

// CHECK-LABEL: @f0(
#[no_mangle]
pub fn f0(stack: &mut Stack, f: fn(&T)) -> bool {
    // CHECK-NOT: icmp eq ptr.*null.*
    f_impl::<0>(stack, f)
}

// CHECK-LABEL: @f1(
#[no_mangle]
pub fn f1(stack: &mut Stack, f: fn(&T)) -> bool {
    // CHECK-NOT: icmp eq ptr.*null.*
    f_impl::<1>(stack, f)
}

// CHECK-LABEL: @f2(
#[no_mangle]
pub fn f2(stack: &mut Stack, f: fn(&T)) -> bool {
    // CHECK-NOT: icmp eq ptr.*null.*
    f_impl::<2>(stack, f)
}

#[inline(always)]
fn f_impl<const N: usize>(stack: &mut Stack, f: fn(&T)) -> bool {
    let Some((a, b)) = stack.popn_top::<N>() else {
        return false;
    };
    a.iter().for_each(f);
    f(b);
    true
}

pub struct Stack {
    data: Vec<T>,
}

impl Stack {
    #[inline]
    fn popn_top<const N: usize>(&mut self) -> Option<([T; N], &mut T)> {
        if self.data.len() < N + 1 {
            return None;
        }
        unsafe { Some((self.popn_unchecked(), self.top_unchecked())) }
    }

    unsafe fn popn_unchecked<const N: usize>(&mut self) -> [T; N] {
        core::array::from_fn(|_| unsafe { self.pop_unchecked() })
    }

    unsafe fn pop_unchecked(&mut self) -> T {
        self.data.pop().unwrap_unchecked()
    }

    unsafe fn top_unchecked(&mut self) -> &mut T {
        self.data.last_mut().unwrap_unchecked()
    }
}
