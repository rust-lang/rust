// edition:2018
// check-pass
#![warn(unused)]
#![allow(unreachable_code)]

pub fn unintentional_copy_one() {
    let mut last = None;
    let mut f = move |s| {
        last = Some(s); //~  WARN value assigned to `last` is never read
                        //~| WARN unused variable: `last`
    };
    f("a");
    f("b");
    f("c");
    dbg!(last.unwrap());
}

pub fn unintentional_copy_two() {
    let mut sum = 0;
    (1..10).for_each(move |x| {
        sum += x; //~ WARN unused variable: `sum`
    });
    dbg!(sum);
}

pub fn f() {
    let mut c = 0;

    // Captured by value, but variable is dead on entry.
    let _ = move || {
        c = 1; //~ WARN value captured by `c` is never read
        println!("{}", c);
    };
    let _ = async move {
        c = 1; //~ WARN value captured by `c` is never read
        println!("{}", c);
    };

    // Read and written to, but never actually used.
    let _ = move || {
        c += 1; //~ WARN unused variable: `c`
    };
    let _ = async move {
        c += 1; //~  WARN value assigned to `c` is never read
                //~| WARN unused variable: `c`
    };

    let _ = move || {
        println!("{}", c);
        // Value is read by closure itself on later invocations.
        c += 1;
    };
    let b = Box::new(42);
    let _ = move || {
        println!("{}", c);
        // Never read because this is FnOnce closure.
        c += 1; //~  WARN value assigned to `c` is never read
        drop(b);
    };
    let _ = async move {
        println!("{}", c);
        // Never read because this is a generator.
        c += 1; //~  WARN value assigned to `c` is never read
    };
}

pub fn nested() {
    let mut d = None;
    let mut e = None;
    let _ = || {
        let _ = || {
            d = Some("d1"); //~ WARN value assigned to `d` is never read
            d = Some("d2");
        };
        let _ = move || {
            e = Some("e1"); //~  WARN value assigned to `e` is never read
                            //~| WARN unused variable: `e`
            e = Some("e2"); //~  WARN value assigned to `e` is never read
        };
    };
}

pub fn g<T: Default>(mut v: T) {
    let _ = |r| {
        if r {
            v = T::default(); //~ WARN value assigned to `v` is never read
        } else {
            drop(v);
        }
    };
}

pub fn h<T: Copy + Default + std::fmt::Debug>() {
    let mut z = T::default();
    let _ = move |b| {
        loop {
            if b {
                z = T::default(); //~  WARN value assigned to `z` is never read
                                  //~| WARN unused variable: `z`
            } else {
                return;
            }
        }
        dbg!(z);
    };
}

fn main() {}
