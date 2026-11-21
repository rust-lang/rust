fn main() {
    let mut nums = vec![2, 3, 5];
    nums.iter_mut().for_each(|&(mut num)| {}); //~ ERROR mismatched types
    nums.iter_mut().for_each(|& /* ( */ num| {}); //~ ERROR mismatched types
    nums.iter_mut().for_each(|& /* ok */ (mut num)| {}); //~ ERROR mismatched types
}
