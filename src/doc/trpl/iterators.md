% Iterators

Let's talk about loops.

Remember Rust's `for` loop? Here's an example:

```{rust}
for x in range(0i, 10i) {
    println!("{}", x);
}
```

Now that you know more Rust, we can talk in detail about how this works. The
`range` function returns an **iterator**. An iterator is something that we can
call the `.next()` method on repeatedly, and it gives us a sequence of things.

Like this:

```{rust}
let mut range = range(0i, 10i);

loop {
    match range.next() {
        Some(x) => {
            println!("{}", x);
        },
        None => { break }
    }
}
```

We make a mutable binding to the return value of `range`, which is our iterator.
We then `loop`, with an inner `match`. This `match` is used on the result of
`range.next()`, which gives us a reference to the next value of the iterator.
`next` returns an `Option<int>`, in this case, which will be `Some(int)` when
we have a value and `None` once we run out. If we get `Some(int)`, we print it
out, and if we get `None`, we `break` out of the loop.

This code sample is basically the same as our `for` loop version. The `for`
loop is just a handy way to write this `loop`/`match`/`break` construct.

`for` loops aren't the only thing that uses iterators, however. Writing your
own iterator involves implementing the `Iterator` trait. While doing that is
outside of the scope of this guide, Rust provides a number of useful iterators
to accomplish various tasks. Before we talk about those, we should talk about a
Rust anti-pattern. And that's `range`.

Yes, we just talked about how `range` is cool. But `range` is also very
primitive. For example, if you needed to iterate over the contents of
a vector, you may be tempted to write this:

```{rust}
let nums = vec![1i, 2i, 3i];

for i in range(0u, nums.len()) {
    println!("{}", nums[i]);
}
```

This is strictly worse than using an actual iterator. The `.iter()` method on
vectors returns an iterator which iterates through a reference to each element
of the vector in turn. So write this:

```{rust}
let nums = vec![1i, 2i, 3i];

for num in nums.iter() {
    println!("{}", num);
}
```

There are two reasons for this. First, this more directly expresses what we
mean. We iterate through the entire vector, rather than iterating through
indexes, and then indexing the vector. Second, this version is more efficient:
the first version will have extra bounds checking because it used indexing,
`nums[i]`. But since we yield a reference to each element of the vector in turn
with the iterator, there's no bounds checking in the second example. This is
very common with iterators: we can ignore unnecessary bounds checks, but still
know that we're safe.

There's another detail here that's not 100% clear because of how `println!`
works. `num` is actually of type `&int`. That is, it's a reference to an `int`,
not an `int` itself. `println!` handles the dereferencing for us, so we don't
see it. This code works fine too:

```{rust}
let nums = vec![1i, 2i, 3i];

for num in nums.iter() {
    println!("{}", *num);
}
```

Now we're explicitly dereferencing `num`. Why does `iter()` give us references?
Well, if it gave us the data itself, we would have to be its owner, which would
involve making a copy of the data and giving us the copy. With references,
we're just borrowing a reference to the data, and so it's just passing
a reference, without needing to do the copy.

So, now that we've established that `range` is often not what you want, let's
talk about what you do want instead.

There are three broad classes of things that are relevant here: iterators,
**iterator adapters**, and **consumers**. Here's some definitions:

* 'iterators' give you a sequence of values.
* 'iterator adapters' operate on an iterator, producing a new iterator with a
  different output sequence.
* 'consumers' operate on an iterator, producing some final set of values.

Let's talk about consumers first, since you've already seen an iterator,
`range`.

## Consumers

A 'consumer' operates on an iterator, returning some kind of value or values.
The most common consumer is `collect()`. This code doesn't quite compile,
but it shows the intention:

```{rust,ignore}
let one_to_one_hundred = range(1i, 101i).collect();
```

As you can see, we call `collect()` on our iterator. `collect()` takes
as many values as the iterator will give it, and returns a collection
of the results. So why won't this compile? Rust can't determine what
type of things you want to collect, and so you need to let it know.
Here's the version that does compile:

```{rust}
let one_to_one_hundred = range(1i, 101i).collect::<Vec<int>>();
```

If you remember, the `::<>` syntax allows us to give a type hint,
and so we tell it that we want a vector of integers.

`collect()` is the most common consumer, but there are others too. `find()`
is one:

```{rust}
let greater_than_forty_two = range(0i, 100i)
                             .find(|x| *x > 42);

match greater_than_forty_two {
    Some(_) => println!("We got some numbers!"),
    None    => println!("No numbers found :("),
}
```

`find` takes a closure, and works on a reference to each element of an
iterator. This closure returns `true` if the element is the element we're
looking for, and `false` otherwise. Because we might not find a matching
element, `find` returns an `Option` rather than the element itself.

Another important consumer is `fold`. Here's what it looks like:

```{rust}
let sum = range(1i, 4i)
              .fold(0i, |sum, x| sum + x);
```

`fold()` is a consumer that looks like this:
`fold(base, |accumulator, element| ...)`. It takes two arguments: the first
is an element called the "base". The second is a closure that itself takes two
arguments: the first is called the "accumulator," and the second is an
"element." Upon each iteration, the closure is called, and the result is the
value of the accumulator on the next iteration. On the first iteration, the
base is the value of the accumulator.

Okay, that's a bit confusing. Let's examine the values of all of these things
in this iterator:

| base | accumulator | element | closure result |
|------|-------------|---------|----------------|
| 0i   | 0i          | 1i      | 1i             |
| 0i   | 1i          | 2i      | 3i             |
| 0i   | 3i          | 3i      | 6i             |

We called `fold()` with these arguments:

```{rust}
# range(1i, 4i)
.fold(0i, |sum, x| sum + x);
```

So, `0i` is our base, `sum` is our accumulator, and `x` is our element.  On the
first iteration, we set `sum` to `0i`, and `x` is the first element of `nums`,
`1i`. We then add `sum` and `x`, which gives us `0i + 1i = 1i`. On the second
iteration, that value becomes our accumulator, `sum`, and the element is
the second element of the array, `2i`. `1i + 2i = 3i`, and so that becomes
the value of the accumulator for the last iteration. On that iteration,
`x` is the last element, `3i`, and `3i + 3i = 6i`, which is our final
result for our sum. `1 + 2 + 3 = 6`, and that's the result we got.

Whew. `fold` can be a bit strange the first few times you see it, but once it
clicks, you can use it all over the place. Any time you have a list of things,
and you want a single result, `fold` is appropriate.

Consumers are important due to one additional property of iterators we haven't
talked about yet: laziness. Let's talk some more about iterators, and you'll
see why consumers matter.

## Iterators

As we've said before, an iterator is something that we can call the
`.next()` method on repeatedly, and it gives us a sequence of things.
Because you need to call the method, this means that iterators
are **lazy** and don't need to generate all of the values upfront.
This code, for example, does not actually generate the numbers
`1-100`, and just creates a value that represents the sequence:

```{rust}
let nums = range(1i, 100i);
```

Since we didn't do anything with the range, it didn't generate the sequence.
Let's add the consumer:

```{rust}
let nums = range(1i, 100i).collect::<Vec<int>>();
```

Now, `collect()` will require that `range()` give it some numbers, and so
it will do the work of generating the sequence.

`range` is one of two basic iterators that you'll see. The other is `iter()`,
which you've used before. `iter()` can turn a vector into a simple iterator
that gives you each element in turn:

```{rust}
let nums = [1i, 2i, 3i];

for num in nums.iter() {
   println!("{}", num);
}
```

These two basic iterators should serve you well. There are some more
advanced iterators, including ones that are infinite. Like `count`:

```{rust}
std::iter::count(1i, 5i);
```

This iterator counts up from one, adding five each time. It will give
you a new integer every time, forever (well, technically, until it reaches the
maximum number representable by an `int`). But since iterators are lazy,
that's okay! You probably don't want to use `collect()` on it, though...

That's enough about iterators. Iterator adapters are the last concept
we need to talk about with regards to iterators. Let's get to it!

## Iterator adapters

"Iterator adapters" take an iterator and modify it somehow, producing
a new iterator. The simplest one is called `map`:

```{rust,ignore}
range(1i, 100i).map(|x| x + 1i);
```

`map` is called upon another iterator, and produces a new iterator where each
element reference has the closure it's been given as an argument called on it.
So this would give us the numbers from `2-100`. Well, almost! If you
compile the example, you'll get a warning:

```{notrust,ignore}
warning: unused result which must be used: iterator adaptors are lazy and
         do nothing unless consumed, #[warn(unused_must_use)] on by default
 range(1i, 100i).map(|x| x + 1i);
 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

Laziness strikes again! That closure will never execute. This example
doesn't print any numbers:

```{rust,ignore}
range(1i, 100i).map(|x| println!("{}", x));
```

If you are trying to execute a closure on an iterator for its side effects,
just use `for` instead.

There are tons of interesting iterator adapters. `take(n)` will return an
iterator over the next `n` elements of the original iterator, note that this
has no side effect on the original iterator. Let's try it out with our infinite
iterator from before, `count()`:

```{rust}
for i in std::iter::count(1i, 5i).take(5) {
    println!("{}", i);
}
```

This will print

```{notrust,ignore}
1
6
11
16
21
```

`filter()` is an adapter that takes a closure as an argument. This closure
returns `true` or `false`. The new iterator `filter()` produces
only the elements that that closure returns `true` for:

```{rust}
for i in range(1i, 100i).filter(|&x| x % 2 == 0) {
    println!("{}", i);
}
```

This will print all of the even numbers between one and a hundred.
(Note that because `filter` doesn't consume the elements that are
being iterated over, it is passed a reference to each element, and
thus the filter predicate uses the `&x` pattern to extract the integer
itself.)

You can chain all three things together: start with an iterator, adapt it
a few times, and then consume the result. Check it out:

```{rust}
range(1i, 1000i)
    .filter(|&x| x % 2 == 0)
    .filter(|&x| x % 3 == 0)
    .take(5)
    .collect::<Vec<int>>();
```

This will give you a vector containing `6`, `12`, `18`, `24`, and `30`.

This is just a small taste of what iterators, iterator adapters, and consumers
can help you with. There are a number of really useful iterators, and you can
write your own as well. Iterators provide a safe, efficient way to manipulate
all kinds of lists. They're a little unusual at first, but if you play with
them, you'll get hooked. For a full list of the different iterators and
consumers, check out the [iterator module documentation](../std/iter/index.html).
