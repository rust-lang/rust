#![warn(clippy::manual_swap)]
#![no_main]

fn swap1() {
    let mut v = [3, 2, 1, 0];
    let index = v[0];
    //~^ manual_swap

    v[0] = v[index];
    v[index] = index;
}

fn swap2() {
    let mut v = [3, 2, 1, 0];
    let tmp = v[0];
    //~^ manual_swap
    v[0] = v[1];
    v[1] = tmp;
    // check not found in this scope.
    let _ = tmp;
}

fn swap3() {
    let mut v = [3, 2];
    let i1 = 0;
    let i2 = 1;
    let temp = v[i1];
    //~^ manual_swap
    v[i1] = v[i2];
    v[i2] = temp;
}

fn swap4() {
    let mut v = [3, 2, 1];
    let i1 = 0;
    let i2 = 1;
    let temp = v[i1];
    //~^ manual_swap
    v[i1] = v[i2 + 1];
    v[i2 + 1] = temp;
}

fn swap5() {
    let mut v = [0, 1, 2, 3];
    let i1 = 0;
    let i2 = 1;
    let temp = v[i1];
    //~^ manual_swap
    v[i1] = v[i2 + 1];
    v[i2 + 1] = temp;
}

fn swap6() {
    let mut v = [0, 1, 2, 3];
    let index = v[0];
    //~^ manual_swap

    v[0] = v[index + 1];
    v[index + 1] = index;
}

fn swap7() {
    let mut v = [0, 1, 2, 3];
    let i1 = 0;
    let i2 = 6;
    let tmp = v[i1 * 3];
    //~^ manual_swap
    v[i1 * 3] = v[i2 / 2];
    v[i2 / 2] = tmp;
}

fn swap8() {
    let mut v = [1, 2, 3, 4];
    let i1 = 1;
    let i2 = 1;
    let tmp = v[i1 + i2];
    //~^ manual_swap
    v[i1 + i2] = v[i2];
    v[i2] = tmp;
}

fn issue_14931() {
    let mut v = [1, 2, 3, 4];

    let mut i1 = 0;
    for i2 in 0..4 {
        let tmp = v[i1];
        //~^ manual_swap
        v[i1] = v[i2];
        v[i2] = tmp;

        i1 += 2;
    }
}
