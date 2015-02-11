- Feature Name: from_elem_with_love
- Start Date: 2015-02-11
- RFC PR: (leave this empty)
- Rust Issue: (leave this empty)

# Summary

Add back `Vec::from_elem`.

# Motivation

High demand, mostly. There are currently a few ways to achieve the behaviour of `Vec::from_elem(elem, n)`:

```
// #1
let vec = Vec::new();
for i in range(0, n) {
    vec.push(elem.clone())
}
```

```
// #2
let vec = vec![elem; n]
```

```
// #3
let vec = Vec::new();
vec.resize(elem, n);
```

```
// #4
let vec: Vec<_> = (0..n).map(|_| elem.clone()).collect()
```

```
// #5
let vec: Vec<_> = iter::repeat(elem).take(n).collect();
```

None of these quite match the convenience, power, and performance of:

```
let vec = Vec::from_elem(elem, n)
```

* `#1` is verbose *and* slow, because each `push` requires a capacity check
* `#2` only works for a Copy `elem` and const `n`
* `#3` needs a temporary, but should be otherwise identical performance-wise.
* `#4` and `#5` suffer from the untrusted iterator len problem performance wise (which is only a temporary 
argument, this will be solved sooner rather than later), and are otherwise verbose and noisy. They also 
need to clone one more time than other methods *strictly* need to.

# Detailed design

Just revert the code deletion.

# Drawbacks

Trivial API bloat, more ways to do the same thing.

# Alternatives

Make the `vec![elem; n]` form more powerful. There's literally no reason the author is aware of 
for the restrictions it has. It would be shorter. It would be cooler. It would give less ways 
to do the same thing. Also it better clarifies the subtle amiguity of what `Vec::from_elem(100, 10)` 
produces. 100 10's? 10 100's?

# Unresolved questions

No.
