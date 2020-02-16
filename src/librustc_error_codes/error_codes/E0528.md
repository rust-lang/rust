An array or slice pattern required more elements than were present in the
matched array.

Example of erroneous code:

```compile_fail,E0528
let r = &[1, 2];
match r {
    &[a, b, c, rest @ ..] => { // error: pattern requires at least 3
                               //        elements but array has 2
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```

Ensure that the matched array has at least as many elements as the pattern
requires. You can match an arbitrary number of remaining elements with `..`:

```
let r = &[1, 2, 3, 4, 5];
match r {
    &[a, b, c, rest @ ..] => { // ok!
        // prints `a=1, b=2, c=3 rest=[4, 5]`
        println!("a={}, b={}, c={} rest={:?}", a, b, c, rest);
    }
}
```
