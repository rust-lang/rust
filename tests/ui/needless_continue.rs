macro_rules! zero {
    ($x:expr) => {
        $x == 0
    };
}

macro_rules! nonzero {
    ($x:expr) => {
        !zero!($x)
    };
}

#[warn(clippy::needless_continue)]
fn main() {
    let mut i = 1;
    while i < 10 {
        i += 1;

        if i % 2 == 0 && i % 3 == 0 {
            println!("{}", i);
            println!("{}", i + 1);
            if i % 5 == 0 {
                println!("{}", i + 2);
            }
            let i = 0;
            println!("bar {} ", i);
        } else {
            continue;
        }

        println!("bleh");
        {
            println!("blah");
        }

        // some comments that also should ideally be included in the
        // output of the lint suggestion if possible.
        if !(!(i == 2) || !(i == 5)) {
            println!("lama");
        }

        if (zero!(i % 2) || nonzero!(i % 5)) && i % 3 != 0 {
            continue;
        } else {
            println!("Blabber");
            println!("Jabber");
        }

        println!("bleh");
    }
}
