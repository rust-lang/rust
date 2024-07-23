//@ run-pass

// Test to make sure the destructors run in the right order.
// Each destructor sets it's tag in the corresponding entry
// in ORDER matching up to when it ran.
// Correct order is: matched, inner, outer


static mut ORDER: [usize; 3] = [0, 0, 0];
static mut INDEX: usize = 0;

struct A;
impl Drop for A {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 1;
            //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
            INDEX = INDEX + 1;
        }
    }
}

struct B;
impl Drop for B {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 2;
            //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
            INDEX = INDEX + 1;
        }
    }
}

struct C;
impl Drop for C {
    fn drop(&mut self) {
        unsafe {
            ORDER[INDEX] = 3;
            //~^ WARN creating a reference to mutable static is discouraged [static_mut_refs]
            INDEX = INDEX + 1;
        }
    }
}

fn main() {
    {
        let matched = A;
        let _outer = C;
        {
            match matched {
                _s => {}
            }
            let _inner = B;
        }
    }
    unsafe {
        let expected: &[_] = &[1, 2, 3];
        assert_eq!(expected, ORDER);
        //~^ WARN creating a shared reference to mutable static is discouraged [static_mut_refs]
    }
}
