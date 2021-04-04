// Test that arrays are completely captured by closures by relying on the borrow check diagnostics

#![feature(capture_disjoint_fields)]
//~^ WARNING: the feature `capture_disjoint_fields` is incomplete

fn arrays_1() {
    let mut arr = [1, 2, 3, 4, 5];

    let mut c = || {
        arr[0] += 10;
    };

    // c will capture `arr` completely, therefore another index into the
    // array can't be modified here
    arr[1] += 10;
    //~^ ERROR: cannot use `arr` because it was mutably borrowed
    //~| ERROR: cannot use `arr[_]` because it was mutably borrowed
    c();
}

fn arrays_2() {
    let mut arr = [1, 2, 3, 4, 5];

    let c = || {
        println!("{:#?}", &arr[3..4]);
    };

    // c will capture `arr` completely, therefore another index into the
    // array can't be modified here
    arr[1] += 10;
    //~^ ERROR: cannot assign to `arr[_]` because it is borrowed
    c();
}

fn arrays_3() {
    let mut arr = [1, 2, 3, 4, 5];

    let c = || {
        println!("{}", arr[3]);
    };

    // c will capture `arr` completely, therefore another index into the
    // array can't be modified here
    arr[1] += 10;
    //~^ ERROR: cannot assign to `arr[_]` because it is borrowed
    c();
}

fn arrays_4() {
    let mut arr = [1, 2, 3, 4, 5];

    let mut c = || {
        arr[1] += 10;
    };

    // c will capture `arr` completely, therefore we cannot borrow another index
    // into the array.
    println!("{}", arr[3]);
    //~^ ERROR: cannot use `arr` because it was mutably borrowed
    //~| ERROR: cannot borrow `arr[_]` as immutable because it is also borrowed as mutable

    c();
}

fn arrays_5() {
    let mut arr = [1, 2, 3, 4, 5];

    let mut c = || {
        arr[1] += 10;
    };

    // c will capture `arr` completely, therefore we cannot borrow other indecies
    // into the array.
    println!("{:#?}", &arr[3..2]);
    //~^ ERROR: cannot borrow `arr` as immutable because it is also borrowed as mutable

    c();
}

fn main() {
    arrays_1();
    arrays_2();
    arrays_3();
    arrays_4();
    arrays_5();
}
