// edition:2021

// Tests that in cases where we individually capture all the fields of a type,
// we still drop them in the order they would have been dropped in the 2018 edition.

// NOTE: It is *critical* that the order of the min capture NOTES in the stderr output
//       does *not* change!

#![feature(rustc_attrs)]

#[derive(Debug)]
struct HasDrop;
impl Drop for HasDrop {
    fn drop(&mut self) {
        println!("dropped");
    }
}

fn test_one() {
    let a = (HasDrop, HasDrop);
    let b = (HasDrop, HasDrop);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
        //~^ ERROR: Min Capture analysis includes:
        //~| ERROR
        println!("{:?}", a.0);
        //~^ NOTE: Min Capture a[(0, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", a.1);
        //~^ NOTE: Min Capture a[(1, 0)] -> ImmBorrow
        //~| NOTE

        println!("{:?}", b.0);
        //~^ NOTE: Min Capture b[(0, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", b.1);
        //~^ NOTE: Min Capture b[(1, 0)] -> ImmBorrow
        //~| NOTE
    };
}

fn test_two() {
    let a = (HasDrop, HasDrop);
    let b = (HasDrop, HasDrop);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
        //~^ ERROR: Min Capture analysis includes:
        //~| ERROR
        println!("{:?}", a.1);
        //~^ NOTE: Min Capture a[(1, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", a.0);
        //~^ NOTE: Min Capture a[(0, 0)] -> ImmBorrow
        //~| NOTE

        println!("{:?}", b.1);
        //~^ NOTE: Min Capture b[(1, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", b.0);
        //~^ NOTE: Min Capture b[(0, 0)] -> ImmBorrow
        //~| NOTE
    };
}

fn test_three() {
    let a = (HasDrop, HasDrop);
    let b = (HasDrop, HasDrop);

    let c = #[rustc_capture_analysis]
    //~^ ERROR: attributes on expressions are experimental
    //~| NOTE: see issue #15701 <https://github.com/rust-lang/rust/issues/15701>
    || {
        //~^ ERROR: Min Capture analysis includes:
        //~| ERROR
        println!("{:?}", b.1);
        //~^ NOTE: Min Capture b[(1, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", a.1);
        //~^ NOTE: Min Capture a[(1, 0)] -> ImmBorrow
        //~| NOTE
        println!("{:?}", a.0);
        //~^ NOTE: Min Capture a[(0, 0)] -> ImmBorrow
        //~| NOTE

        println!("{:?}", b.0);
        //~^ NOTE: Min Capture b[(0, 0)] -> ImmBorrow
        //~| NOTE
    };
}

fn main() {
    test_one();
    test_two();
    test_three();
}
