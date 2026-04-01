// Test that we error if a slice is modified after it has been bounds checked
// and before we actually index it.

fn modify_before_assert_slice_slice(x: &[&[i32]]) -> i32 {
    let mut x = x;
    let z: &[i32] = &[1, 2, 3];
    let y: &[&[i32]] = &[z];
    x[{ x = y; 0 }][2]              // OK we haven't checked any bounds before we index `x`.
}

fn modify_before_assert_array_slice(x: &[&[i32]; 3]) -> i32 {
    let mut x = x;
    let z: &[i32] = &[1, 2, 3];
    let y: &[&[i32]; 3] = &[z, z, z];
    x[{ x = y; 0 }][2]              // OK we haven't checked any bounds before we index `x`.
}

fn modify_before_assert_slice_array(x: &[&[i32; 3]]) -> i32 {
    let mut x = x;
    let z: &[i32; 3] = &[1, 2, 3];
    let y: &[&[i32; 3]] = &[z];
    x[{ x = y; 0 }][2]              // OK we haven't checked any bounds before we index `x`.
}

fn modify_before_assert_array_array(x: &[&[i32; 3]; 3]) -> i32 {
    let mut x = x;
    let z: &[i32; 3] = &[1, 2, 3];
    let y: &[&[i32; 3]; 3] = &[z, z, z];
    x[{ x = y; 0 }][2]              // OK we haven't checked any bounds before we index `x`.
}

fn modify_after_assert_slice_slice(x: &[&[i32]]) -> i32 {
    let mut x = x;
    let z: &[i32] = &[1, 2, 3];
    let y: &[&[i32]] = &[&z];
    x[1][{ x = y; 2}]               //~ ERROR cannot assign `x` in indexing expression
}

fn modify_after_assert_array_slice(x: &[&[i32]; 1]) -> i32 {
    let mut x = x;
    let z: &[i32] = &[1, 2, 3];
    let y: &[&[i32]; 1] = &[&z];
    x[0][{ x = y; 2}]               // OK cannot invalidate a fixed-size array bounds check
}

fn modify_after_assert_slice_array(x: &[&[i32; 3]]) -> i32 {
    let mut x = x;
    let z: &[i32; 3] = &[1, 2, 3];
    let y: &[&[i32; 3]] = &[&z];
    x[1][{ x = y; 2}]               //~ ERROR cannot assign `x` in indexing expression
}

fn modify_after_assert_array_array(x: &[&[i32; 3]; 1]) -> i32 {
    let mut x = x;
    let z: &[i32; 3] = &[1, 2, 3];
    let y: &[&[i32; 3]; 1] = &[&z];
    x[0][{ x = y; 2}]               // OK cannot invalidate a fixed-size array bounds check
}

fn modify_after_assert_slice_slice_array(x: &[&[[i32; 1]]]) -> i32 {
    let mut x = x;
    let z: &[[i32; 1]] = &[[1], [2], [3]];
    let y: &[&[[i32; 1]]] = &[&z];
    x[1][{ x = y; 2}][0]            //~ ERROR cannot assign `x` in indexing expression
}

fn modify_after_assert_slice_slice_slice(x: &[&[&[i32]]]) -> i32 {
    let mut x = x;
    let z: &[&[i32]] = &[&[1], &[2], &[3]];
    let y: &[&[&[i32]]] = &[z];
    x[1][{ x = y; 2}][0]            //~ ERROR cannot assign `x` in indexing expression
}


fn main() {
    println!("{}", modify_after_assert_slice_array(&[&[4, 5, 6], &[9, 10, 11]]));
    println!("{}", modify_after_assert_slice_slice(&[&[4, 5, 6], &[9, 10, 11]]));
    println!("{}", modify_after_assert_slice_slice_array(&[&[[4], [5], [6]], &[[9], [10], [11]]]));
    println!("{}", modify_after_assert_slice_slice_slice(
        &[&[&[4], &[5], &[6]], &[&[9], &[10], &[11]]]),
    );
}
