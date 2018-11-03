fn make_vec() -> Vec<u8> {
    let mut v = Vec::with_capacity(4);
    v.push(1);
    v.push(2);
    v
}

fn make_vec_macro() -> Vec<u8> {
    vec![1, 2]
}

fn make_vec_macro_repeat() -> Vec<u8> {
    vec![42; 5]
}

fn make_vec_macro_repeat_zeroed() -> Vec<u8> {
    vec![0; 7]
}

fn vec_into_iter() -> u8 {
    vec![1, 2, 3, 4]
        .into_iter()
        .map(|x| x * x)
        .fold(0, |x, y| x + y)
}

fn vec_into_iter_rev() -> u8 {
    vec![1, 2, 3, 4]
        .into_iter()
        .map(|x| x * x)
        .fold(0, |x, y| x + y)
}

fn vec_into_iter_zst() -> usize {
    vec![[0u64; 0], [0u64; 0]]
        .into_iter()
        .rev()
        .map(|x| x.len())
        .sum()
}

fn vec_into_iter_rev_zst() -> usize {
    vec![[0u64; 0], [0u64; 0]]
        .into_iter()
        .rev()
        .map(|x| x.len())
        .sum()
}

fn vec_iter_and_mut() {
    let mut v = vec![1,2,3,4];
    for i in v.iter_mut() {
        *i += 1;
    }
    assert_eq!(v.iter().sum::<i32>(), 2+3+4+5);
}

fn vec_iter_and_mut_rev() {
    let mut v = vec![1,2,3,4];
    for i in v.iter_mut().rev() {
        *i += 1;
    }
    assert_eq!(v.iter().sum::<i32>(), 2+3+4+5);
}

fn vec_reallocate() -> Vec<u8> {
    let mut v = vec![1, 2];
    v.push(3);
    v.push(4);
    v.push(5);
    v
}

fn main() {
    assert_eq!(vec_reallocate().len(), 5);

    assert_eq!(vec_into_iter(), 30);
    assert_eq!(vec_into_iter_rev(), 30);
    vec_iter_and_mut();
    assert_eq!(vec_into_iter_zst(), 0);
    assert_eq!(vec_into_iter_rev_zst(), 0);
    vec_iter_and_mut_rev();

    assert_eq!(make_vec().capacity(), 4);
    assert_eq!(make_vec_macro(), [1, 2]);
    assert_eq!(make_vec_macro_repeat(), [42; 5]);
    assert_eq!(make_vec_macro_repeat_zeroed(), [0; 7]);
}
