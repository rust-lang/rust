#![warn(clippy::while_let_on_iterator)]
#![allow(dead_code, unreachable_code, unused_mut)]
#![allow(
    clippy::equatable_if_let,
    clippy::manual_find,
    clippy::never_loop,
    clippy::redundant_closure_call,
    clippy::single_range_in_vec_init,
    clippy::uninlined_format_args,
    clippy::useless_vec
)]

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

fn refutable2() {
    // Issue 3780
    {
        let v = vec![1, 2, 3];
        let mut it = v.windows(2);
        while let Some([x, y]) = it.next() {
            println!("x: {}", x);
            println!("y: {}", y);
        }

        let mut it = v.windows(2);
        while let Some([x, ..]) = it.next() {
            println!("x: {}", x);
        }

        let mut it = v.windows(2);
        while let Some([.., y]) = it.next() {
            println!("y: {}", y);
        }

        let mut it = v.windows(2);
        while let Some([..]) = it.next() {}

        let v = vec![[1], [2], [3]];
        let mut it = v.iter();
        while let Some([1]) = it.next() {}

        let mut it = v.iter();
        while let Some([_x]) = it.next() {}
    }

    // binding
    {
        let v = vec![1, 2, 3];
        let mut it = v.iter();
        while let Some(x @ 1) = it.next() {
            println!("{}", x);
        }

        let v = vec![[1], [2], [3]];
        let mut it = v.iter();
        while let Some(x @ [_]) = it.next() {
            println!("{:?}", x);
        }
    }

    // false negative
    {
        let v = vec![1, 2, 3];
        let mut it = v.iter().map(Some);
        while let Some(Some(_) | None) = it.next() {
            println!("1");
        }
    }
}

fn nested_loops() {
    let a = [42, 1337];

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
    // This should not cause an ICE

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

fn issue6491() {
    // Used in outer loop, needs &mut
    let mut it = 1..40;
    while let Some(n) = it.next() {
        while let Some(m) = it.next() {
            if m % 10 == 0 {
                break;
            }
            println!("doing something with m: {}", m);
        }
        println!("n still is {}", n);
    }

    // This is fine, inner loop uses a new iterator.
    let mut it = 1..40;
    while let Some(n) = it.next() {
        let mut it = 1..40;
        while let Some(m) = it.next() {
            if m % 10 == 0 {
                break;
            }
            println!("doing something with m: {}", m);
        }

        // Weird binding shouldn't change anything.
        let (mut it, _) = (1..40, 0);
        while let Some(m) = it.next() {
            if m % 10 == 0 {
                break;
            }
            println!("doing something with m: {}", m);
        }

        // Used after the loop, needs &mut.
        let mut it = 1..40;
        while let Some(m) = it.next() {
            if m % 10 == 0 {
                break;
            }
            println!("doing something with m: {}", m);
        }
        println!("next item {}", it.next().unwrap());

        println!("n still is {}", n);
    }
}

fn issue6231() {
    // Closure in the outer loop, needs &mut
    let mut it = 1..40;
    let mut opt = Some(0);
    while let Some(n) = opt.take().or_else(|| it.next()) {
        while let Some(m) = it.next() {
            if n % 10 == 0 {
                break;
            }
            println!("doing something with m: {}", m);
        }
        println!("n still is {}", n);
    }
}

fn issue1924() {
    struct S<T>(T);
    impl<T: Iterator<Item = u32>> S<T> {
        fn f(&mut self) -> Option<u32> {
            // Used as a field.
            while let Some(i) = self.0.next() {
                if !(3..8).contains(&i) {
                    return Some(i);
                }
            }
            None
        }

        fn f2(&mut self) -> Option<u32> {
            // Don't lint, self borrowed inside the loop
            while let Some(i) = self.0.next() {
                if i == 1 {
                    return self.f();
                }
            }
            None
        }
    }
    impl<T: Iterator<Item = u32>> S<(S<T>, Option<u32>)> {
        fn f3(&mut self) -> Option<u32> {
            // Don't lint, self borrowed inside the loop
            while let Some(i) = self.0.0.0.next() {
                if i == 1 {
                    return self.0.0.f();
                }
            }
            while let Some(i) = self.0.0.0.next() {
                if i == 1 {
                    return self.f3();
                }
            }
            // This one is fine, a different field is borrowed
            while let Some(i) = self.0.0.0.next() {
                if i == 1 {
                    return self.0.1.take();
                } else {
                    self.0.1 = Some(i);
                }
            }
            None
        }
    }

    struct S2<T>(T, u32);
    impl<T: Iterator<Item = u32>> Iterator for S2<T> {
        type Item = u32;
        fn next(&mut self) -> Option<u32> {
            self.0.next()
        }
    }

    // Don't lint, field of the iterator is accessed in the loop
    let mut it = S2(1..40, 0);
    while let Some(n) = it.next() {
        if n == it.1 {
            break;
        }
    }

    // Needs &mut, field of the iterator is accessed after the loop
    let mut it = S2(1..40, 0);
    while let Some(n) = it.next() {
        if n == 0 {
            break;
        }
    }
    println!("iterator field {}", it.1);
}

fn issue7249() {
    let mut it = 0..10;
    let mut x = || {
        // Needs &mut, the closure can be called multiple times
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    };
    x();
    x();
}

fn issue7510() {
    let mut it = 0..10;
    let it = &mut it;
    // Needs to reborrow `it` as the binding isn't mutable
    while let Some(x) = it.next() {
        if x % 2 == 0 {
            break;
        }
    }
    println!("{}", it.next().unwrap());

    struct S<T>(T);
    let mut it = 0..10;
    let it = S(&mut it);
    // Needs to reborrow `it.0` as the binding isn't mutable
    while let Some(x) = it.0.next() {
        if x % 2 == 0 {
            break;
        }
    }
    println!("{}", it.0.next().unwrap());
}

fn exact_match_with_single_field() {
    struct S<T>(T);
    let mut s = S(0..10);
    // Don't lint. `s.0` is used inside the loop.
    while let Some(_) = s.0.next() {
        let _ = &mut s.0;
    }
}

fn custom_deref() {
    struct S1<T> {
        x: T,
    }
    struct S2<T>(S1<T>);
    impl<T> core::ops::Deref for S2<T> {
        type Target = S1<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl<T> core::ops::DerefMut for S2<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    let mut s = S2(S1 { x: 0..10 });
    while let Some(x) = s.x.next() {
        println!("{}", x);
    }
}

fn issue_8113() {
    let mut x = [0..10];
    while let Some(x) = x[0].next() {
        println!("{}", x);
    }
}

fn fn_once_closure() {
    let mut it = 0..10;
    (|| {
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    })();

    fn f(_: impl FnOnce()) {}
    let mut it = 0..10;
    f(|| {
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    });

    fn f2(_: impl FnMut()) {}
    let mut it = 0..10;
    f2(|| {
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    });

    fn f3(_: fn()) {}
    f3(|| {
        let mut it = 0..10;
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    });

    trait MySpecialFnMut: FnOnce() {}
    impl<T: FnOnce()> MySpecialFnMut for T {}
    fn f4(_: impl MySpecialFnMut) {}
    let mut it = 0..10;
    f4(|| {
        while let Some(x) = it.next() {
            if x % 2 == 0 {
                break;
            }
        }
    });
}

fn issue13123() {
    let mut it = 0..20;
    'label: while let Some(n) = it.next() {
        if n % 25 == 0 {
            break 'label;
        }
    }
}

fn main() {
    let mut it = 0..20;
    while let Some(..) = it.next() {
        println!("test");
    }
}
