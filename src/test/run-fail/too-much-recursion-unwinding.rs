// xfail-test leaks
// error-pattern:ran out of stack

// Test that the task fails after hiting the recursion limit
// during unwinding

fn recurse() {
    log(debug, "don't optimize me out");
    recurse();
}

struct r {
    recursed: *mut bool,
    drop unsafe { 
        if !*(self.recursed) {
            *(self.recursed) = true;
            recurse();
        }
    }
}

fn r(recursed: *mut bool) -> r unsafe {
    r { recursed: recursed }
}

fn main() {
    let mut recursed = false;
    let _r = r(ptr::mut_addr_of(&recursed));
    recurse();
}