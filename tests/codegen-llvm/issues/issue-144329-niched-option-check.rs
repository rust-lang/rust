//! Ensure that redundant null checks on `&mut T` from `Option<(_, &mut T)>` are eliminated.

//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

type T = [u64; 4];

// CHECK-LABEL: @f(
#[no_mangle]
pub fn f(stack: &mut Stack, f: fn(&T)) -> bool {
    // CHECK-NOT: icmp eq ptr
    let Some((_a, b)) = stack.popn_top::<0>() else {
        return false;
    };
    f(b);
    true
}

pub struct Stack {
    data: Vec<T>,
}

impl Stack {
    #[inline(always)]
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
