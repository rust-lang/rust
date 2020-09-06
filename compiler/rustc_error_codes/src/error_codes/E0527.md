The number of elements in an array or slice pattern differed from the number of
elements in the array being matched.

Example of erroneous code:

```compile_fail,E0527
let r = &[1, 2, 3, 4];
match r {
    &[a, b] => { // error: pattern requires 2 elements but array
                 //        has 4
        println!("a={}, b={}", a, b);
    }
}
```

Ensure that the pattern is consistent with the size of the matched
array. Additional elements can be matched with `..`:

```
let r = &[1, 2, 3, 4];
match r {
    &[a, b, ..] => { // ok!
        println!("a={}, b={}", a, b);
    }
}
```
