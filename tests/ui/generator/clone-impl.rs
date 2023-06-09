// gate-test-generator_clone
// Verifies that non-static generators can be cloned/copied if all their upvars and locals held
// across awaits can be cloned/copied.

#![feature(generators, generator_clone)]

struct NonClone;

fn main() {
    let copyable: u32 = 123;
    let clonable_0: Vec<u32> = Vec::new();
    let clonable_1: Vec<u32> = Vec::new();
    let non_clonable: NonClone = NonClone;

    let gen_copy_0 = move || {
        yield;
        drop(copyable);
    };
    check_copy(&gen_copy_0);
    check_clone(&gen_copy_0);
    let gen_copy_1 = move || {
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
    let gen_clone_0 = move || {
        let v = vec!['a'];
        yield;
        drop(v);
        drop(clonable_0);
    };
    check_copy(&gen_clone_0);
    //~^ ERROR the trait bound `Vec<u32>: Copy` is not satisfied
    //~| ERROR the trait bound `Vec<char>: Copy` is not satisfied
    check_clone(&gen_clone_0);
    let gen_clone_1 = move || {
        let v = vec!['a'];
        /*
        let n = NonClone;
        drop(n);
        */
        yield;
        let n = NonClone;
        drop(n);
        drop(v);
        drop(clonable_1);
    };
    check_copy(&gen_clone_1);
    //~^ ERROR the trait bound `Vec<u32>: Copy` is not satisfied
    //~| ERROR the trait bound `Vec<char>: Copy` is not satisfied
    check_clone(&gen_clone_1);
    let gen_non_clone = move || {
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
