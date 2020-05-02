// run-rustfix

#![warn(clippy::while_let_on_iterator)]
#![allow(clippy::never_loop, unreachable_code, unused_mut)]

fn base() {
    let mut iter = 1..20;
    while let Option::Some(x) = iter.next() {
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(x) = iter.next() {
        println!("{}", x);
    }

    let mut iter = 1..20;
    while let Some(_) = iter.next() {}

    let mut iter = 1..20;
    while let None = iter.next() {} // this is fine (if nonsensical)

    let mut iter = 1..20;
    if let Some(x) = iter.next() {
        // also fine
        println!("{}", x)
    }

    // the following shouldn't warn because it can't be written with a for loop
    let mut iter = 1u32..20;
    while let Some(_) = iter.next() {
        println!("next: {:?}", iter.next())
    }

    // neither can this
    let mut iter = 1u32..20;
    while let Some(_) = iter.next() {
        println!("next: {:?}", iter.next());
    }

    // or this
    let mut iter = 1u32..20;
    while let Some(_) = iter.next() {
        break;
    }
    println!("Remaining iter {:?}", iter);

    // or this
    let mut iter = 1u32..20;
    while let Some(_) = iter.next() {
        iter = 1..20;
    }
}

// Issue #1188
fn refutable() {
    let a = [42, 1337];
    let mut b = a.iter();

    // consume all the 42s
    while let Some(&42) = b.next() {}

    let a = [(1, 2, 3)];
    let mut b = a.iter();

    while let Some(&(1, 2, 3)) = b.next() {}

    let a = [Some(42)];
    let mut b = a.iter();

    while let Some(&None) = b.next() {}

    /* This gives “refutable pattern in `for` loop binding: `&_` not covered”
    for &42 in b {}
    for &(1, 2, 3) in b {}
    for &Option::None in b.next() {}
    // */
}

fn nested_loops() {
    let a = [42, 1337];
    let mut y = a.iter();
    loop {
        // x is reused, so don't lint here
        while let Some(_) = y.next() {}
    }

    let mut y = a.iter();
    for _ in 0..2 {
        while let Some(_) = y.next() {
            // y is reused, don't lint
        }
    }

    loop {
        let mut y = a.iter();
        while let Some(_) = y.next() {
            // use a for loop here
        }
    }
}

fn issue1121() {
    use std::collections::HashSet;
    let mut values = HashSet::new();
    values.insert(1);

    while let Some(&value) = values.iter().next() {
        values.remove(&value);
    }
}

fn issue2965() {
    // This should not cause an ICE and suggest:
    //
    // for _ in values.iter() {}
    //
    use std::collections::HashSet;
    let mut values = HashSet::new();
    values.insert(1);

    while let Some(..) = values.iter().next() {}
}

fn issue3670() {
    let array = [Some(0), None, Some(1)];
    let mut iter = array.iter();

    while let Some(elem) = iter.next() {
        let _ = elem.or_else(|| *iter.next()?);
    }
}

fn issue1654() {
    // should not lint if the iterator is generated on every iteration
    use std::collections::HashSet;
    let mut values = HashSet::new();
    values.insert(1);

    while let Some(..) = values.iter().next() {
        values.remove(&1);
    }

    while let Some(..) = values.iter().map(|x| x + 1).next() {}

    let chars = "Hello, World!".char_indices();
    while let Some((i, ch)) = chars.clone().next() {
        println!("{}: {}", i, ch);
    }
}

fn main() {
    base();
    refutable();
    nested_loops();
    issue1121();
    issue2965();
    issue3670();
    issue1654();
}
