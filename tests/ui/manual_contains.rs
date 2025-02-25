#![warn(clippy::manual_contains)]
#![allow(clippy::eq_op, clippy::useless_vec)]

fn should_lint() {
    let vec: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let _ = values.iter().any(|&v| v == 4);
    //~^ manual_contains

    let vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let _ = values.iter().any(|&v| v == 4);
    //~^ manual_contains

    let values: [u8; 6] = [3, 14, 15, 92, 6, 5];
    let _ = values.iter().any(|&v| v == 10);
    //~^ manual_contains

    let num = 14;
    let values: [u8; 6] = [3, 14, 15, 92, 6, 5];
    let _ = values.iter().any(|&v| v == num);
    //~^ manual_contains

    let num = 14;
    let values: [u8; 6] = [3, 14, 15, 92, 6, 5];
    let _ = values.iter().any(|&v| num == v);
    //~^ manual_contains

    let values: [u8; 6] = [3, 14, 15, 92, 6, 5];
    let _ = values.iter().any(|v| *v == 4);
    //~^ manual_contains

    let vec: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let _ = values.iter().any(|&v| 4 == v);
    //~^ manual_contains

    let vec: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let a = &4;
    let _ = values.iter().any(|v| *v == *a);
    //~^ manual_contains

    let vec = vec!["1", "2", "3", "4", "5", "6"];
    let values = &vec[..];
    let _ = values.iter().any(|&v| v == "4");
    //~^ manual_contains

    let vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let _ = values.iter().any(|&v| v == 4 + 1);
    //~^ manual_contains
}

fn should_not_lint() {
    let values: [u8; 6] = [3, 14, 15, 92, 6, 5];
    let _ = values.iter().any(|&v| v > 10);

    let vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let _ = values.iter().any(|&v| v.is_multiple_of(2));
    let _ = values.iter().any(|&v| v * 2 == 6);
    let _ = values.iter().any(|&v| v == v);
    let _ = values.iter().any(|&v| 4 == 4);
    let _ = values.contains(&4);

    let a = 1;
    let b = 2;
    let _ = values.iter().any(|&v| a == b);
    let _ = values.iter().any(|&v| a == 4);

    let vec: Vec<String> = vec!["1", "2", "3", "4", "5", "6"]
        .iter()
        .map(|&x| x.to_string())
        .collect();
    let values = &vec[..];
    let _ = values.iter().any(|v| v == "4");

    let vec: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let values = &vec[..];
    let mut counter = 0;
    let mut count = || {
        counter += 1;
        counter
    };
    let _ = values.iter().any(|&v| v == count());
    let _ = values.iter().any(|&v| v == v * 2);
}

fn foo(values: &[u8]) -> bool {
    values.iter().any(|&v| v == 10)
    //~^ manual_contains
}

fn bar(values: [u8; 3]) -> bool {
    values.iter().any(|&v| v == 10)
    //~^ manual_contains
}
