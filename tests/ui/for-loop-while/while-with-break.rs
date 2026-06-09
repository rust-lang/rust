//@ run-pass

pub fn main() {
    let mut i: isize = 90;
    while i < 100 {
        println!("{}", i);
        i = i + 1;
        if i == 95 {
            let _v: Vec<isize> =
                vec![1, 2, 3, 4, 5]; // we check that it is freed by break

            println!("breaking");
            break;
        }
    }
    assert_eq!(i, 95);
}
