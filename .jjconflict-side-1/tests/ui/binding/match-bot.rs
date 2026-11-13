//@ run-pass

pub fn main() {
    let i: isize =
        match Some::<isize>(3) { None::<isize> => { panic!() } Some::<isize>(_) => { 5 } };
    println!("{}", i);
}
