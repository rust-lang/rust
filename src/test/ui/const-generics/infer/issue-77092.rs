use std::convert::TryInto;

fn take_array_from_mut<T, const N: usize>(data: &mut [T], start: usize) -> &mut [T; N] {
    (&mut data[start .. start + N]).try_into().unwrap()
}

fn main() {
    let mut arr = [0, 1, 2, 3, 4, 5, 6, 7, 8];

    for i in 1 .. 4 {
        println!("{:?}", take_array_from_mut(&mut arr, i));
        //~^ ERROR type annotations needed
    }
}
