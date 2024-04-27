// Regression test for issue #72649
// Tests that we don't emit spurious
// 'value moved in previous iteration of loop' message

struct NonCopy;
//~^ NOTE if `NonCopy` implemented `Clone`
//~| NOTE if `NonCopy` implemented `Clone`
//~| NOTE if `NonCopy` implemented `Clone`
//~| NOTE if `NonCopy` implemented `Clone`
//~| NOTE consider implementing `Clone` for this type
//~| NOTE consider implementing `Clone` for this type
//~| NOTE consider implementing `Clone` for this type
//~| NOTE consider implementing `Clone` for this type

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
        //~| NOTE you could clone this value
        let _used2 = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value used here after move
    }
}

fn moved_here_2() {
    let value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    loop { //~ NOTE inside of this loop
        let _used = value;
        //~^ NOTE value moved here
        //~| NOTE you could clone this value
        loop {
            let _used2 = value; //~ ERROR use of moved value: `value`
            //~^ NOTE value used here after move
        }
    }
}

fn moved_loop_1() {
    let value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    loop { //~ NOTE inside of this loop
        let _used = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value moved here, in previous iteration of loop
        //~| NOTE you could clone this value
    }
}

fn moved_loop_2() {
    let mut value = NonCopy{};
    //~^ NOTE move occurs because `value` has type `NonCopy`, which does not implement the `Copy` trait
    let _used = value;
    value = NonCopy{};
    loop { //~ NOTE inside of this loop
        let _used2 = value; //~ ERROR use of moved value: `value`
        //~^ NOTE value moved here, in previous iteration of loop
        //~| NOTE you could clone this value
    }
}

fn uninit_1() {
    loop {
        let value: NonCopy; //~ NOTE declared here
        let _used = value; //~ ERROR binding `value` isn't initialized
        //~^ NOTE `value` used here but it isn't initialized
    }
}

fn uninit_2() {
    let mut value: NonCopy; //~ NOTE declared here
    loop {
        let _used = value; //~ ERROR binding `value` isn't initialized
        //~^ NOTE `value` used here but it isn't initialized
    }
}

fn main() {}
