// gate-test-coroutine_clone
// Verifies that non-static coroutines can be cloned/copied if all their upvars and locals held
// across awaits can be cloned/copied.
//@compile-flags: --diagnostic-width=300

#![feature(coroutines, coroutine_clone, stmt_expr_attributes)]

struct NonClone;

fn test1() {
    let copyable: u32 = 123;
    let gen_copy_0 = #[coroutine]
    move || {
        yield;
        drop(copyable);
    };
    check_copy(&gen_copy_0);
    check_clone(&gen_copy_0);
}

fn test2() {
    let copyable: u32 = 123;
    let gen_copy_1 = #[coroutine]
    move || {
        /*
        let v = vec!['a'];
        let n = NonClone;
        drop(v);
        drop(n);
        */
        yield;
        let v = vec!['a'];
        let n = NonClone;
        drop(n);
        drop(copyable);
    };
    check_copy(&gen_copy_1);
    check_clone(&gen_copy_1);
}

fn test3_upvars() {
    let clonable_0: Vec<u32> = Vec::new();
    let gen_clone_0 = #[coroutine]
    move || {
        yield;
        drop(clonable_0);
    };
    check_copy(&gen_clone_0);
    //~^ ERROR the trait bound `Vec<u32>: Copy` is not satisfied
    check_clone(&gen_clone_0);
}

fn test3_witness() {
    let gen_clone_1 = #[coroutine]
    move || {
        let v = vec!['a'];
        yield;
        drop(v);
    };
    check_copy(&gen_clone_1);
    //~^ ERROR the trait bound `Vec<char>: Copy` is not satisfied
    check_clone(&gen_clone_1);
}

fn test4() {
    let clonable_1: Vec<u32> = Vec::new();
    let gen_clone_1 = #[coroutine]
    move || {
        yield;
        let n = NonClone;
        drop(n);
        drop(clonable_1);
    };
    check_copy(&gen_clone_1);
    //~^ ERROR the trait bound `Vec<u32>: Copy` is not satisfied
    check_clone(&gen_clone_1);
}

fn test5() {
    let non_clonable: NonClone = NonClone;
    let gen_non_clone = #[coroutine]
    move || {
        yield;
        drop(non_clonable);
    };
    check_copy(&gen_non_clone);
    //~^ ERROR the trait bound `NonClone: Copy` is not satisfied
    check_clone(&gen_non_clone);
    //~^ ERROR the trait bound `NonClone: Clone` is not satisfied
}

fn check_copy<T: Copy>(_x: &T) {}
fn check_clone<T: Clone>(_x: &T) {}

fn main() {}
