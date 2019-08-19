fn main() {
    let _foo = &[1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `&[usize; 2]` as `[usize]`

    let _bar = Box::new(1_usize) as dyn std::fmt::Debug;
    //~^ ERROR cast to unsized type: `std::boxed::Box<usize>` as `dyn std::fmt::Debug`

    let _baz = 1_usize as dyn std::fmt::Debug;
    //~^ ERROR cast to unsized type: `usize` as `dyn std::fmt::Debug`

    let _quux = [1_usize, 2] as [usize];
    //~^ ERROR cast to unsized type: `[usize; 2]` as `[usize]`
}
