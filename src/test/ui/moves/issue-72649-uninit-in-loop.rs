// Regression test for issue #72649
// Tests that we don't emit spurious
// 'value moved in previous iteration of loop' message

struct NonCopy;

fn good() {
    loop {
        let value = NonCopy{};
        let _used = value;
    }
}

fn moved_here_1() {
    loop {
        let value = NonCopy{};
        //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
        let _used = value;
        //~^ NOTE value moved here
        let _used2 = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value used here after move
    }
}

fn moved_here_2() {
    let value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    loop {
        let _used = value;
        //~^ NOTE value moved here
        loop {
            let _used2 = value; //~ ERROR use of moved value: `value`
            //~^ NOTE value used here after move
        }
    }
}

fn moved_loop_1() {
    let value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    loop {
        let _used = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value moved here, in previous iteration of loop
    }
}

fn moved_loop_2() {
    let mut value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    let _used = value;
    value = NonCopy{};
    loop {
        let _used2 = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value moved here, in previous iteration of loop
    }
}

fn uninit_1() {
    loop {
        let value: NonCopy;
        let _used = value; //~ ERROR use of possibly-uninitialized variable: `value`
        //~^ NOTE use of possibly-uninitialized `value`
    }
}

fn uninit_2() {
    let mut value: NonCopy;
    loop {
        let _used = value; //~ ERROR use of possibly-uninitialized variable: `value`
        //~^ NOTE use of possibly-uninitialized `value`
    }
}

fn main() {}
