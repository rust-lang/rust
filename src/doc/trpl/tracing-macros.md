% Tracing Macros

The `trace_macros` feature allows you to use a special feature: tracing macro
invocations.

In the advanced macros chapter, we defined a `bct` macro:

```rust
macro_rules! bct {
    // cmd 0:  d ... => ...
    (0, $($ps:tt),* ; $_d:tt)
        => (bct!($($ps),*, 0 ; ));
    (0, $($ps:tt),* ; $_d:tt, $($ds:tt),*)
        => (bct!($($ps),*, 0 ; $($ds),*));

    // cmd 1p:  1 ... => 1 ... p
    (1, $p:tt, $($ps:tt),* ; 1)
        => (bct!($($ps),*, 1, $p ; 1, $p));
    (1, $p:tt, $($ps:tt),* ; 1, $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; 1, $($ds),*, $p));

    // cmd 1p:  0 ... => 0 ...
    (1, $p:tt, $($ps:tt),* ; $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; $($ds),*));

    // halt on empty data string
    ( $($ps:tt),* ; )
        => (());
}
```

This is pretty complex! we can see the output

 ```rust
#![feature(trace_macros)]

macro_rules! bct {
    // cmd 0:  d ... => ...
    (0, $($ps:tt),* ; $_d:tt)
        => (bct!($($ps),*, 0 ; ));
    (0, $($ps:tt),* ; $_d:tt, $($ds:tt),*)
        => (bct!($($ps),*, 0 ; $($ds),*));

    // cmd 1p:  1 ... => 1 ... p
    (1, $p:tt, $($ps:tt),* ; 1)
        => (bct!($($ps),*, 1, $p ; 1, $p));
    (1, $p:tt, $($ps:tt),* ; 1, $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; 1, $($ds),*, $p));

    // cmd 1p:  0 ... => 0 ...
    (1, $p:tt, $($ps:tt),* ; $($ds:tt),*)
        => (bct!($($ps),*, 1, $p ; $($ds),*));

    // halt on empty data string
    ( $($ps:tt),* ; )
        => (());
}

fn main() {
    trace_macros!(true);

    bct!(0, 0, 1, 1, 1 ; 1, 0, 1);
}

This will print out a wall of text:

```text
bct! { 0 , 0 , 1 , 1 , 1 ; 1 , 0 , 1 }
bct! { 0 , 1 , 1 , 1 , 0 ; 0 , 1 }
bct! { 1 , 1 , 1 , 0 , 0 ; 1 }
bct! { 1 , 0 , 0 , 1 , 1 ; 1 , 1 }
bct! { 0 , 1 , 1 , 1 , 0 ; 1 , 1 , 0 }
bct! { 1 , 1 , 1 , 0 , 0 ; 1 , 0 }
bct! { 1 , 0 , 0 , 1 , 1 ; 1 , 0 , 1 }
bct! { 0 , 1 , 1 , 1 , 0 ; 1 , 0 , 1 , 0 }
bct! { 1 , 1 , 1 , 0 , 0 ; 0 , 1 , 0 }
bct! { 1 , 0 , 0 , 1 , 1 ; 0 , 1 , 0 }
bct! { 0 , 1 , 1 , 1 , 0 ; 0 , 1 , 0 }
```

And eventually, error:

```text
18:45 error: recursion limit reached while expanding the macro `bct`
    => (bct!($($ps),*, 1, $p ; $($ds),*));
        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

The `trace_macros!` call is what produces this output, showing how we match
each time.
